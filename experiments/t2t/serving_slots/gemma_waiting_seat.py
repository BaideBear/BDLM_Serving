"""One discretionary Q64 warm MASK seat; last physical projection is metadata, never logits."""
from .gemma_joint_verify_eval import check, digest


def replacement(rows, dirty, accepted, stamps):
    """All row ids are local. Keep the frontier and all existing prefix/debt seats."""
    rows, dirty, accepted = set(rows), set(dirty), set(accepted)
    masks = set(range(256)) - accepted
    check(len(stamps) == 256 and min(stamps) >= 0 and rows <= set(range(256))
          and dirty <= rows and masks, "Not a fully initialized sampled head")
    outside = masks - rows
    expendable = (rows & masks) - dirty - {min(masks)}
    if not outside or not expendable:
        return None
    drop = max(expendable)
    take = min(outside, key=lambda r: (stamps[r], r))
    return (drop, take) if stamps[take] < stamps[drop] else None


def arm_class(base_class, enabled, receipts):
    class WaitingSeat(base_class):
        def _apply(self, states, pos, sample, prefix, cold, offer, route):
            check(offer == 64 and len(states) == 1 and self.tail is None and not self.rebuild,
                  "Fixed Q64 solo only")
            h, = states
            r = h.rows
            base = self.origin + 256*h.ordinal
            planned = tuple(pos)
            warm = route == "sparse" and bool(sample) and len(r.initialized) == 256
            note = dict(tick=self.tick, head=h.ordinal, warm=warm, contested=False, swap=None)
            if warm:
                check(not cold and h.can_sample and set(sample) == set(pos) and len(h.sampler.sc)==256,
                      "Wrong warm phase")
                local = {x-base for x in pos}
                masks = set(range(256)) - r.accepted
                clean = local & masks - r.dirty
                check(not (set(prefix) & {base+x for x in masks | r.dirty}), "Invalid prefix")
                note["contested"] = bool(masks-local and clean-{min(masks)})
                pair = replacement(local, r.dirty, r.accepted, r._stamps)
                if pair:
                    d, a = pair
                    alternative = tuple(sorted((set(pos)-{base+d})|{base+a}))
                    check(len(pos)==len(alternative)==64 and set(prefix)<=set(alternative)
                          and r.dirty <= {x-base for x in alternative}, "Seat swap touched reserved work")
                    note["swap"] = dict(drop=d, take=a, drop_stamp=r._stamps[d], take_stamp=r._stamps[a],
                                        age=r.age, frontier=min(masks), width=len(pos),
                                        state=digest([h.ordinal, r.age, sorted(r.accepted), sorted(r.dirty),
                                                      r._stamps, planned, sorted(prefix)]),
                                        reference_rows=digest(planned), alternative_rows=digest(alternative))
                    if enabled:
                        pos, sample = alternative, set(alternative)
            e = super()._apply(states, pos, sample, prefix, cold, offer, route)
            check(tuple(e["positions"]) == tuple(pos), "Executor did not project the requested plan")
            if warm:
                p, = e["parts"]
                check([base+x for x in p["sampled"]] == list(pos) and
                      {base+x for x in p["prefix"]}==set(prefix), "Changed sampler width or prefix")
            receipts.append(note)
            return e
    return WaitingSeat


def alignment(off, on, shadows):
    a,b = shadows
    check(len(a)==len(off["tape"]) and len(b)==len(on["tape"]), "Missing seat receipt")
    i = next((j for j,x in enumerate(b) if x["swap"]), None)
    if i is None:
        check(a==b and off==on, "Unexplained divergence without an applied swap")
        return None
    check(a[:i+1] == b[:i+1] and off["tape"][:i]==on["tape"][:i]
          and off["tape"][i]!=on["tape"][i], "First different forward not the registered seat")
    return dict(**b[i], predecessor_sha256=digest(off["tape"][:i]))


def gates(rows):
    check(tuple(r["index"] for r in rows)==(0,10,20), "Wrong anchor set")
    return dict(exposure=any(r["first"] for r in rows),
                no_losses=all(not r["off"]["correct"] or r["on"]["correct"] for r in rows),
                one_F_saving=any(r["on"]["F"]<r["off"]["F"] for r in rows),
                total_F_not_higher=sum(r["on"]["F"]-r["off"]["F"] for r in rows)<=0)
