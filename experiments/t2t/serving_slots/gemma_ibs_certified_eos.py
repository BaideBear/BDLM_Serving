"""Opt-in EOS stop for the pinned, single-block hard-capped Gemma IBS loop."""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

BASELINE_SHA = "a09d04477361cb2a260dac85ce2195c0bfee9218306b7cfb96d7ddf4f1dba9af"
B, CAP = 256, 2048
OFFERS = (32, 64, 128)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def eligible(ready, eos_index, debt, offer):
    """Global absolute debt, irrespective of what the next planner would pick."""
    if eos_index is None or not 0 <= eos_index < ready < B:
        return None
    if not debt:
        return "clean"
    if len(debt) <= offer and all(eos_index < x < B for x in debt):
        return "suffix_debt"
    return None


class Stop:
    def __init__(self, arm, offer):
        require(arm in {"control", "strict", "paid"}, "Unknown EOS arm")
        require(type(offer) is int and offer in OFFERS, "Explicit registered offer required")
        self.arm = arm
        self.offer = offer
        self.first_clean = self.first_paid = self.trigger = self.payment = None
        self.head = None
        self.ready = 0
        self.eos_index = self.eos_id = None

    def rows(self, normal, prior, dirty, start, offer):
        require(offer == self.offer, "IBS planner offer differs from the registered offer")
        if self.trigger is None or self.trigger["kind"] != "suffix_debt":
            return normal
        import numpy as np
        debt = self.trigger["debt"]
        local = np.asarray(debt, dtype=np.int64) - start
        require(self.payment is None and self.trigger["block"] * B == start
                and 0 < len(local) <= offer
                and np.flatnonzero(dirty).tolist() == debt
                and prior[local].all() and np.isin(local, normal).all(),
                "Not the single registered suffix-only payment")
        return local

    def after_update(self, ids, resolved, dirty, start, ready, eos, trace):
        """Invoked after the real update_from_mask and physical trace append."""
        import numpy as np
        turn = trace[-1]
        require(turn["offer"] == self.offer, "IBS trace offer differs from the registered offer")
        debt = np.flatnonzero(dirty).tolist()
        if self.trigger is not None:
            hit = self.trigger
            require(hit["kind"] == "suffix_debt" and turn["tick"] == hit["tick"] + 1
                    and turn["positions"] == hit["debt"] and not debt and not turn["admitted"]
                    and not turn["accepted"] and start // B == hit["block"],
                    "Payment changed admissions, lost debt or advanced to another block")
            self.payment = dict(tick=turn["tick"], positions=turn["positions"], Q=turn["q"])
            return ids[0, :start + hit["eos_index"] + 1].tolist()

        if start != self.head:
            self.head, self.ready = start, 0
            self.eos_index = self.eos_id = None
        require(self.ready <= ready <= B and resolved[start:start + ready].all()
                and all(not start <= x < start + ready for x in debt),
                "IBS frontier is not an immutable, paid local prefix")
        if self.eos_index is None and ready > self.ready and ready < B:
            frozen = ids[0, start + self.ready:start + ready].tolist()
            if (i := next((i for i, token in enumerate(frozen) if token in eos), None)) is not None:
                self.eos_index, self.eos_id = self.ready + i, frozen[i]
        self.ready = ready
        kind = eligible(ready, self.eos_index, [x-start for x in debt], turn["offer"])
        if kind is None:
            return None
        cert = dict(kind=kind, tick=turn["tick"], block=start // B, ready=ready,
                    eos_index=self.eos_index, eos_id=self.eos_id, debt=debt)
        if self.first_paid is None:
            self.first_paid = cert
        if kind == "clean" and self.first_clean is None:
            self.first_clean = cert
        if self.arm == "control" or (self.arm == "strict" and kind != "clean"):
            return None
        self.trigger = cert
        if kind == "clean":
            return ids[0, :start + self.eos_index + 1].tolist()
        return None


_PLAN = "            n = len(local)\n"
_AFTER = "            if qualify and len(trace)==2:\n"
_SIG = "def _generate(bank, model, data, controls, unmask, mode, c, qualify, caps):"


@lru_cache(maxsize=1)
def _instrumented_loop(rt):
    # Exact-source gate: no fork of the decoder and no monkeypatch of its globals.
    path = Path(rt.__file__)
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == BASELINE_SHA, "Unqualified IBS runtime source")
    src = raw.decode().split(_SIG)
    require(len(src) == 2, "Missing IBS loop")
    body = src[1].split("\n\ndef ", 1)[0]
    require(body.count(_PLAN) == body.count(_AFTER) == 1, "Ambiguous IBS hook anchors")
    require("_ibs_eos" not in body, "EOS adapter name collides with stock locals")
    body = body.replace(_PLAN, "            local = _ibs_eos.rows(local, prior, dirty, start, offer)\n" + _PLAN)
    body = body.replace(_AFTER, """\
            delivery = _ibs_eos.after_update(ids, resolved, dirty, start, ctl.cached_prefix_len, eos, trace)
            if delivery is not None:
                return delivery, trace
""" + _AFTER)
    namespace = {}
    source = "\n" * src[0].count("\n") + _SIG[:-2] + ", _ibs_eos):" + body
    exec(compile(source, f"<certified-eos {path}>", "exec"), vars(rt), namespace)
    return namespace["_generate"]


def generate(rt, model, data, controls, unmask, stop):
    loop = _instrumented_loop(rt)
    bank = rt.Resident()
    bank.bypass = False
    with rt.resident_attention(model, bank):
        return loop(bank, model, data, controls, unmask, "ibs", stop.offer, False, (stop.offer,), stop)
