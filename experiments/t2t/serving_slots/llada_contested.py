"""One saturated warm head seat: optional decoded prefix versus resident MASK."""
from dataclasses import replace

from .llada_fixed_family_eval import require


def exchange(plan, sampled, masks, resident, q):
    if (not plan.prefix or plan.cold or plan.drain_only or len(plan.rows) != q
            or sampled != plan.rows):
        return plan, sampled, None
    pool = sorted((masks & resident) - set(plan.rows))
    if not pool:
        return plan, sampled, None
    drop, take = min(plan.prefix), pool[0]
    require(drop < min(masks) <= take and drop // 32 == take // 32
            and drop not in plan.dirty, "not clean head prefix vs waiting MASK")
    rows = tuple(sorted((set(plan.rows) - {drop}) | {take}))
    return replace(plan, rows=rows, prefix=tuple(p for p in plan.prefix if p != drop),
                   mask=tuple(sorted((*plan.mask, take)))), rows, (drop, take)


def variant(control):
    class Contested(control):
        def _cross_plan(self, q):
            p, s, phase, cap = super()._cross_plan(q)
            self.contest = None
            if p.prefix and self.cross == "head" and not self.borrow:
                masks, _, _ = self._inputs(32)
                p, s, self.contest = exchange(p, s, masks, self.bank.initialized, q)
            return p, s, phase, cap

        def _step(self, q):
            e = super()._step(q)
            if self.contest:
                e["contest"] = self.contest
            return e
    return Contested


def analyze(a, b, q, mask_id):
    decoded, first, swaps, accepts = set(), None, 0, 0
    for i, y in enumerate(b["trace"]):
        if "contest" in y:
            drop, take = y["contest"]
            st = y["pre_state"]
            lo, hi = max(st["origin"], st["head"]), min(st["end"], st["head"]+32)
            left = next(p for p in range(lo, hi) if p not in decoded)
            pre_rows = (set(y["rows"]) - {take}) | {drop}
            pool = sorted(set(range(lo, min(st["last_end"], hi))) - decoded - pre_rows)
            require(lo <= drop < left <= take < hi and drop in decoded and drop not in y["dirty"]
                    and all(p > drop for p in y["prefix"]) and pool and take == pool[0]
                    and take in y["mask"] and y["sampled"] == y["rows"]
                    and y["proposals"][y["sampled"].index(take)][0] == mask_id
                    and not y["cold"] and y["q"] == q and y["layer_physical_q"] == [q]*20,
                    "wrong same-width replacement")
            swaps += 1
            accepts += take in y["admitted"]
        if first is None:
            require(i < len(a["trace"]), "extra call before exposure")
            x = a["trace"][i]
            if "contest" in y:
                require(all(x[k] == y[k] for k in ("pre_state","cold","dirty","phase","visible_kv",
                            "sample_clock","q","layer_physical_q")) and x["sampled"] == x["rows"]
                        and set(x["rows"]) == pre_rows and set(y["mask"]) == set(x["mask"]) | {take}
                        and set(x["prefix"]) == set(y["prefix"]) | {drop}
                        and x["proposals"][x["sampled"].index(drop)][0] != mask_id,
                        "additional first-state change")
                first = dict(tick=i, drop=drop, take=take, q=q)
            else:
                require(x == y, "different before exchange")
        decoded.update(y["admitted"])
        decoded.difference_update(y["revoked"])
    if not first:
        require(len(a["trace"]) == len(b["trace"]) and all(a[k] == b[k] for k in
                ("output_ids","score","failure","Q","N","F")), "unexposed outcome changed")
    return dict(first=first, calls=swaps, swapped_seats=swaps, mask_admits=accepts, extra_Q=0)
