"""Gemma strict IBS + certified paid-EOS two-observation agreement adapter.

The shared ``EvidenceLedger`` owns the vote rule. This module only adapts the
pinned IBS loop: it reuses the certified-EOS ``rows``/``after_update`` hooks and
adds one read/apply hook around the existing IBS unmasker call. The observer
arm never changes the base sampler result; the apply arm sets the extra
accepted local positions before the stock ``take``/noise write, so the RNG draw
shape and all other stock rules are unchanged and the added rows enter the
original admission debt.
"""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import torch

from bdlm_serving.methods.t2t.paid_agreement import EvidenceLedger
from .gemma_ibs_certified_eos import BASELINE_SHA, B, require


class Agreement:
    """One ledger per request; only measured unresolved rows are observed."""

    def __init__(self, *, apply, excluded_ids, floor=0.5):
        self.ledger = EvidenceLedger(floor=floor)
        self.apply = bool(apply)
        self.excluded_ids = set(excluded_ids)
        self.record_take = None
        self.observations = []
        self.first_extra = None

    def observe(self, unmask, raw, proposal, prior, sel, width, start, tick):
        updated = unmask.low_confidence_dynamic_sparse(
            raw, proposal, torch.as_tensor(prior.copy(), device=raw.device),
            sel, row_width=width)
        positions = (start + sel).tolist()
        probabilities = torch.softmax(raw.float(), dim=-1).gather(
            -1, proposal.unsqueeze(-1)).squeeze(-1).tolist()
        native = updated[sel].bool().tolist()
        extra = self.ledger.observe(
            block=start // B, tick=tick, positions=positions,
            proposals=proposal.tolist(), probabilities=probabilities,
            native=native, excluded_ids=self.excluded_ids)
        self.observations.append(dict(
            tick=tick, block=start // B, positions=positions,
            proposals=proposal.tolist(), probabilities=probabilities,
            native=native, extra=list(extra)))
        if self.apply and extra:
            for pos in extra:
                updated[pos - start] = 1
            if self.record_take is not None:
                self.record_take(updated[sel].bool().tolist())
            if self.first_extra is None:
                self.first_extra = dict(tick=tick, block=start // B, extra=list(extra))
        return updated


_PLAN = "            n = len(local)\n"
_AFTER = "            if qualify and len(trace)==2:\n"
_UNMASK = """                updated = unmask.low_confidence_dynamic_sparse(raw, proposal,
                    torch.as_tensor(prior.copy(),device=device),sel,row_width=width)
"""
_SIG = "def _generate(bank, model, data, controls, unmask, mode, c, qualify, caps):"


@lru_cache(maxsize=1)
def _instrumented_loop(rt):
    path = Path(rt.__file__)
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == BASELINE_SHA, "Unqualified IBS runtime source")
    src = raw.decode().split(_SIG)
    require(len(src) == 2, "Missing IBS loop")
    body = src[1].split("\n\ndef ", 1)[0]
    require(body.count(_PLAN) == body.count(_AFTER) == body.count(_UNMASK) == 1,
            "Ambiguous IBS hook anchors")
    require("_ibs_eos" not in body and "_ibs_agreement" not in body,
            "adapter name collides with stock locals")
    body = body.replace(_PLAN, "            local = _ibs_eos.rows(local, prior, dirty, start, offer)\n" + _PLAN)
    body = body.replace(_AFTER, """\
            delivery = _ibs_eos.after_update(ids, resolved, dirty, start, ctl.cached_prefix_len, eos, trace)
            if delivery is not None:
                return delivery, trace
""" + _AFTER)
    body = body.replace(_UNMASK, """\
                updated = _ibs_agreement.observe(unmask, raw, proposal, prior, sel, width, start, tick)
""")
    namespace = {}
    source = "\n" * src[0].count("\n") + _SIG[:-2] + ", _ibs_eos, _ibs_agreement):" + body
    exec(compile(source, f"<paid-agreement {path}>", "exec"), vars(rt), namespace)
    return namespace["_generate"]


def generate(rt, model, data, controls, unmask, stop, agreement):
    loop = _instrumented_loop(rt)
    bank = rt.Resident()
    bank.bypass = False
    with rt.resident_attention(model, bank):
        return loop(bank, model, data, controls, unmask, "ibs", stop.offer, False,
                    (stop.offer,), stop, agreement)
