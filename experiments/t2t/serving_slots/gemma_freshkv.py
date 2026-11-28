"""A0 Gemma Q64 paid-IBS: same-state fresh-visible-KV counterfactual.

Read-only diagnostic. At the first non-first ordinary sampling step of the
first physical block that re-samples a row already projected but still an
unresolved MASK, re-project the exact legal visible KV interval [start, end)
through the resident attention *bypassed* (fresh) bank on the same canvas IDs,
self-conditioning, encoder cache and absolute positions. Then compare the same
selected-unresolved rows' top-1, p>=.9 crossing and native unmask decision
(including the guaranteed fallback row) against the real resident sparse
output of that tick.

It never rewrites KV, IDs, RNG or the sampler: the scratch forward runs with
``bank.bypass=True`` (no ``extend``/``visited`` writes) and all tracer hooks are
suppressed while it runs, so the whole-request output / F / Q and native
sampling decisions are unchanged. Extra diagnostic forwards/rows are reported
separately (``extra_F`` / ``extra_Q``) and are not part of the paid bill.

Boundaries (do not over-read):
  * This is a same-width comparison: the fresh attention width is exactly the
    resident visible width ``end-start`` (never an artificial B=256 expansion).
  * For a selected row that was already projected, ``sc_bank`` holds the actual
    pre-tick self-conditioning the real call used. For visible rows that are
    not re-selected this tick, the stored ``sc_bank`` value is the only
    available self-conditioning; that part is flagged, not claimed exact.
  * A warm row's stale K also feeds every other query in the interval, so any
    disagreement here is not by itself proof of pure cache error.
"""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from . import gemma_paid_agreement as rep
from .gemma_ibs_certified_eos import BASELINE_SHA, B, Stop, require


def _floats(t):
    return [round(float(x), 6) for x in t.detach().float().cpu().tolist()]


class FreshKV:
    """One-shot per request; fires at the registered same-state tick."""

    def __init__(self, hold):
        self.hold = hold
        self.first_sampling_tick = None
        self.fired = False
        self.record = None
        self.sampled = set()   # absolute positions previously sampled in block 0

    def observe(self, *, logits, prior, local, width, start, ids, cache, masks, plen, kw,
                sc_bank, seen, bank, device, tick, model, unmask):
        if self.fired:
            return
        if start // B != 0:            # first physical block only
            return
        unresolved = prior[local] == 0
        if not bool(unresolved.any()):  # drain / non-sampling tick
            return
        sel_abs = (start + local[unresolved])
        sel_abs_list = sel_abs.tolist()
        if self.first_sampling_tick is None:   # the first sampling step is excluded
            self.first_sampling_tick = tick
            self.sampled.update(sel_abs_list)
            return
        if tick <= self.first_sampling_tick:
            self.sampled.update(sel_abs_list)
            return
        # Registered trigger: a selected-unresolved row that was already *sampled*
        # (not merely projected) at an earlier tick and is still a MASK.
        prev_sampled = np.isin(sel_abs, list(self.sampled))
        if not bool(prev_sampled.any()):
            self.sampled.update(sel_abs_list)
            return

        self.fired = True
        end = int(bank.end)
        bwidth = end - start
        # select() just marked the current rows initialized; restore pre-tick seen.
        seen_full = bank.initialized[start:end].copy()
        seen_full[local] = seen
        idx_full = torch.arange(start, end, device=device)
        pos_full = (idx_full + plen).to(torch.int32)[None]
        sc_full = sc_bank[idx_full % (2 * B)][None] if seen_full.any() else None

        def fresh_forward():
            args = dict(decoder_input_ids=ids[:, idx_full],
                        self_conditioning_logits=sc_full,
                        decoder_attention_mask=masks,
                        decoder_position_ids=pos_full,
                        past_key_values=cache, **kw)
            from contextlib import ExitStack
            with ExitStack() as st:
                if sc_full is not None and not seen_full.all():
                    valid = torch.as_tensor(seen_full, device=device)[None, :, None]
                    h = model.model.decoder.self_conditioning.register_forward_pre_hook(
                        lambda _, a: (a[0], a[1] * valid))
                    st.callback(h.remove)
                bank.bypass = True
                try:
                    return model(**args).logits[0]
                finally:
                    bank.bypass = False

        self.hold['suppress'] = True
        try:
            fresh1 = fresh_forward()
            fresh2 = fresh_forward()
        finally:
            self.hold['suppress'] = False
        repeat_equal = bool(torch.equal(fresh1, fresh2))
        repeat_max = float((fresh1.float() - fresh2.float()).abs().max().item())

        sel_np = local[unresolved]                       # offsets within the block
        sel_t = torch.as_tensor(sel_np, device=device)
        raw = logits[torch.as_tensor(unresolved, device=device)]
        fresh_sel = fresh1[sel_t]

        def decide(x):
            prop = x.argmax(-1)
            upd = unmask.low_confidence_dynamic_sparse(
                x, prop, torch.as_tensor(prior.copy(), device=device), sel_t, row_width=width)
            return prop, upd[sel_t].bool()

        self.hold['suppress'] = True
        try:
            p_raw, take_raw = decide(raw)
            p_fresh, take_fresh = decide(fresh_sel)
        finally:
            self.hold['suppress'] = False

        s_raw = raw.float().softmax(-1).gather(-1, p_raw.unsqueeze(-1)).squeeze(-1)
        s_fresh = fresh_sel.float().softmax(-1).gather(-1, p_fresh.unsqueeze(-1)).squeeze(-1)
        logit_diff = (raw.float() - fresh_sel.float()).abs().amax(-1)

        warm_flags = seen[unresolved]            # projected before (K refreshed if selected)
        sampled_flags = prev_sampled             # sampled before (registered trigger)
        # Unrefreshed warm-unresolved positions in [start,end): stale K would feed queries.
        unresolved_full = prior[:bwidth] == 0
        selected_mask = np.zeros(bwidth, dtype=bool)
        selected_mask[local] = True
        unrefreshed = unresolved_full & seen_full & ~selected_mask
        positions = (start + sel_np).tolist()
        rows = []
        for k, pos in enumerate(positions):
            rows.append(dict(
                pos=int(pos), warm=bool(warm_flags[k]),
                top1_raw=int(p_raw[k]), top1_fresh=int(p_fresh[k]),
                score_raw=round(float(s_raw[k]), 6), score_fresh=round(float(s_fresh[k]), 6),
                p9_raw=bool(float(s_raw[k]) >= .9), p9_fresh=bool(float(s_fresh[k]) >= .9),
                take_raw=bool(take_raw[k]), take_fresh=bool(take_fresh[k]),
                logit_maxabs=round(float(logit_diff[k]), 6)))

        def count(field, warm_only=False):
            return sum(1 for r in rows if r[field] and (r["warm"] or not warm_only))

        top1_flips = [r["pos"] for r in rows if r["top1_raw"] != r["top1_fresh"]]
        p9_flips = [r["pos"] for r in rows if r["p9_raw"] != r["p9_fresh"]]
        take_flips = [r["pos"] for r in rows if r["take_raw"] != r["take_fresh"]]
        warm_rows = [r for r in rows if r["warm"]]
        fb_raw = int(sel_np[int(torch.argmax(s_raw).item())])
        fb_fresh = int(sel_np[int(torch.argmax(s_fresh).item())])
        self.record = dict(
            fired=True, tick=int(tick), block=0, start=int(start), end=int(end),
            row_width=int(width), visible_width=int(bwidth), selected=int(len(local)),
            unresolved=int(len(sel_np)), warm_unresolved=int(len(warm_rows)),
            sampled_prev=int(sampled_flags.sum()),
            unrefreshed_warm=int(unrefreshed.sum()),
            refresh_full=bool(int(unrefreshed.sum()) == 0),
            new_unresolved=int(len(sel_np) - len(warm_rows)),
            repeat_equal=repeat_equal, repeat_maxabs=round(repeat_max, 6),
            maxabs_logit_diff=round(float(logit_diff.max().item()), 6),
            warm_maxabs_logit_diff=(round(float(max((r["logit_maxabs"] for r in warm_rows),
                                                    default=0.0)), 6)),
            top1_flips=top1_flips, p9_flips=p9_flips, take_flips=take_flips,
            fallback_raw=int(start + fb_raw), fallback_fresh=int(start + fb_fresh),
            fallback_equal=bool(fb_raw == fb_fresh),
            extra_F=2, extra_Q=2 * bwidth, rows=rows,
            sc_note=("selected rows use exact pre-tick sc_bank; other visible rows use "
                     "their stored sc_bank masked to pre-tick initialized"),
            same_width=True)


@lru_cache(maxsize=1)
def _loop(rt):
    raw = Path(rt.__file__).read_bytes()
    require(sha256(raw).hexdigest() == BASELINE_SHA, "Unqualified IBS runtime source")
    src = raw.decode().split(rep._SIG)
    require(len(src) == 2, "Missing IBS loop")
    body = src[1].split("\n\ndef ", 1)[0]
    logits = "                logits = model(**args).logits[0]\n"
    anchors = (rep._PLAN, rep._AFTER, rep._UNMASK, logits)
    require(all(body.count(a) == 1 for a in anchors), "Ambiguous pinned-loop hook anchors")
    require("_ibs_fresh" not in body, "fresh hook name collides with stock locals")
    body = body.replace(rep._PLAN, "            local = _ibs_eos.rows(local, prior, dirty, start, offer)\n" + rep._PLAN)
    body = body.replace(rep._AFTER, """\
            delivery = _ibs_eos.after_update(ids, resolved, dirty, start, ctl.cached_prefix_len, eos, trace)
            if delivery is not None:
                return delivery, trace
""" + rep._AFTER)
    body = body.replace(rep._UNMASK, """\
                updated = _ibs_agreement.observe(unmask, raw, proposal, prior, sel, width, start, tick)
""")
    body = body.replace(logits, logits + (
        "                _ibs_fresh.observe(logits=logits, prior=prior, local=local, width=width,\n"
        "                    start=start, ids=ids, cache=cache, masks=masks, plen=plen, kw=kw,\n"
        "                    sc_bank=sc_bank, seen=seen, bank=bank, device=device, tick=tick,\n"
        "                    model=model, unmask=unmask)\n"))
    namespace = {}
    source = "\n" * src[0].count("\n") + rep._SIG[:-2] + ", _ibs_eos, _ibs_agreement, _ibs_fresh):" + body
    exec(compile(source, f"<freshkv {Path(rt.__file__)}>", "exec"), vars(rt), namespace)
    return namespace["_generate"]


def probe_generate(rt, model, data, controls, unmask, stop, agreement, fresh):
    loop = _loop(rt)
    bank = rt.Resident()
    bank.bypass = False
    with rt.resident_attention(model, bank):
        return loop(bank, model, data, controls, unmask, "ibs", stop.offer, False,
                    (stop.offer,), stop, agreement, fresh)
