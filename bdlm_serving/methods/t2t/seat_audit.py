"""Request-local instrumentation for a one-seat intervention; not a serving policy."""
import hashlib
import json
from .context_gate import ADMISSION, ONCE


MODES = ("hard_once_base", "hard_once_rank", "hard_once_random")
GEMMA_CONDITIONAL = "hard_once_mask"
GEMMA_FACTORIAL = tuple("hard_once_" + k + s for k in "kd" for s in "kd")
_KEY = dict(zip(MODES, ("base", "targeted", "random")))
_KEY.update(zip(ONCE, ("base", "targeted", "random")))
_KEY.update(dict.fromkeys(ADMISSION, "targeted"))
_KEY[GEMMA_CONDITIONAL] = "conditional_mask"


def digest(*items):
    """Length-framed SHA-256 of tensor bytes (including dtype/shape) and JSON metadata."""
    import torch
    out = hashlib.sha256()

    def add(buf):
        out.update(len(buf).to_bytes(8, "big"))
        out.update(buf)

    def visit(value):
        if torch.is_tensor(value):
            add(b"tensor")
            add(str((str(value.dtype), tuple(value.shape))).encode())
            buf = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
            add(memoryview(buf))
        elif isinstance(value, (tuple, list)):
            add(b"sequence")
            add(str(len(value)).encode())
            for v in value:
                visit(v)
        else:
            add(b"scalar")
            add(json.dumps(value, separators=(",", ":"), sort_keys=True).encode())
    visit(items)
    return out.hexdigest()


def canvas_digest(ids, resolved, *, accepted_only=False):
    if accepted_only:
        ids = [t if r else None for t, r in zip(ids, resolved, strict=True)]
    return hashlib.sha256(json.dumps([ids, resolved], separators=(",", ":")).encode()).hexdigest()


class SeatAudit:
    def __init__(self, mode):
        self.mode = _KEY[mode]
        self.used = False
        self.current = None
        self.watch_block = None
        self.watch_rows = ()

    def observe_masks(self, event, block, output_origin, ids, resolved):
        """Record accepted IDs after forward; never expose drafts to the selector."""
        if self.current is not None:
            self.watch_block = block
            self.watch_rows = tuple(sorted(set(self.current["base"]).intersection(self.current["masks"])))
        if self.watch_block != block:
            return
        event["beneficiary"] = {
            "rows": list(self.watch_rows),
            "ids": [ids[i] if resolved[i] else None for i in self.watch_rows],
            "output_offsets": [output_origin + i for i in self.watch_rows],
        }

    def choose(self, base, eligible, resolved, fresh, writes, queries, clock, budget,
               model, seed, strict):
        if self.used:
            return base, []
        from .intervention import plan_seat
        plan = plan_seat(eligible, resolved, fresh, writes, queries, clock, budget,
                         model, seed, previous_strict=strict)
        return self.apply_plan(base, plan)

    def apply_plan(self, base, plan):
        if self.used:
            raise ValueError("One-seat intervention already spent")
        if plan is None:
            return base, []
        if list(base) != plan["base"]:
            raise ValueError("One-seat control differs from physical noedit selection")
        key = self.mode
        if key == "conditional_mask":
            key = "targeted" if plan["stratum"] == "mask" else "base"
        picked = plan[key]
        self.used = True
        self.current = dict(plan, executed=picked)
        return picked, sorted(set(picked) - set(base))

    def begin(self, pre_hash, kv, offset):
        p = self.current
        p["pre_sha256"] = pre_hash
        rows = sorted({p["victim"], p["candidate"], p["random_candidate"]})
        idx = [offset + i for i in rows]
        self.before = (rows, idx, [t[..., idx, :].detach().float() for t in kv])

    def end(self, event, ids, resolved, prior_ids, prior_resolved, kv):
        p = self.current
        rows, idx, before = self.before
        p["after_sha256"] = canvas_digest(ids, resolved)
        p["after_admissions"] = sum(not a and b for a, b in zip(prior_resolved, resolved))
        victim = p["victim"]
        p["victim_queried"] = victim in p["executed"]
        p["victim_unresolved_before"] = not prior_resolved[victim]
        p["victim_accepted"] = bool(not prior_resolved[victim] and resolved[victim])
        if p["victim_accepted"] and not p["victim_queried"]:
            raise ValueError("Skipped victim became accepted")
        p["strict_rows"] = [i for i in rows if resolved[i] and prior_resolved[i]
                            and ids[i] != prior_ids[i]]
        p["revoked_rows"] = [i for i in rows if prior_resolved[i] and not resolved[i]]
        drift = []
        for a, t in zip(before, kv, strict=True):
            b = t[..., idx, :].float()
            dims = tuple(i for i in range(a.ndim) if i != a.ndim - 2)
            # Relative to the actual stale pre-intervention resident vector.
            drift.append(((b-a).square().sum(dims).sqrt() /
                          a.square().sum(dims).sqrt().clamp_min(1e-12)).tolist())
        p["resident_l15"] = [dict(row=i, dK=k, dV=v) for i, k, v in zip(rows, *drift, strict=True)]
        if any((r["dK"] or r["dV"]) and r["row"] not in p["executed"] for r in p["resident_l15"]):
            raise ValueError("A skipped KV row changed")
        event["intervention"] = p
        self.current = self.before = None
