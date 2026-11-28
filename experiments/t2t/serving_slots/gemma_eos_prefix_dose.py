"""Instrument the existing fixed verified-EOS request; the sole setting is Law.den."""
from bdlm_serving.methods.t2t.budget_family import Law, choose
from .gemma_joint_verify_eval import check, digest


ROLES = ("dirty", "mask", "prefix", "verify_clean")


def classify(rows, debt, unresolved, prefix, route):
    rows, debt, unresolved, prefix = map(set, (rows, debt, unresolved, prefix))
    verify = route in ("verify", "verify_paid")
    check(debt <= rows and prefix <= rows and not prefix & (debt | unresolved)
          and not (verify and prefix), "Invalid physical row classes")
    paid = rows & debt
    mask = set() if verify else (rows-paid) & unresolved
    closing = rows-paid if verify else set()
    check(paid | mask | prefix | closing == rows, "Unclassified physical rows")
    return dict(zip(ROLES, map(len, (paid, mask, prefix, closing))))


def arm_class(base_cls, den, observations):
    check(den in (4, 8), "Unregistered prefix dose")

    class Dose(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            check(self.fixed_q == 64 and self.law == Law(1, 8) and self.early and
                  self.tick == 0 and self.top_gate == "off" and self.fuse_first_verify and
                  self.ablation == "00" and self.prefix_order == "right" and not self.reuse_paid_masks,
                  "Wrong base profile")
            self.law = Law(1, den)
            self.rows.law = self.law

        def _apply(self, states, pos, sample, prefix, cold, offer, route):
            h, = states
            check(offer == 64 and self.tail is None and not self.rebuild and
                  route in ("sparse", "drain", "verify", "verify_paid"), "Wrong dose route")
            base = self.origin + 256*h.ordinal
            local = tuple(p-base for p in pos)
            pref = {p-base for p in prefix}
            dirty = set(h.rows.dirty)
            mask = set(range(256))-h.rows.accepted
            init = set(h.rows.initialized)
            counts = classify(local, dirty, mask, pref, route)
            note = dict(tick=self.tick, head=h.ordinal, route=route, width=len(local), **counts,
                        verify_dirty=counts["dirty"] if route.startswith("verify") else 0,
                        cold_mask=len((set(local)-dirty) & (mask-init)) if route=="sparse" else 0,
                        more_prefix=0, fewer_masks=0, extra_width=0)
            if route == "sparse":
                # Both plans are metadata-only counterfactuals on this exact state.
                pool = [i for i in reversed(range(min(mask, default=256))) if i in init-dirty]
                plans = {d:choose(64, 256, dirty, mask, pool, law=Law(1,d),
                                  cold=set(range(256))-init) for d in (8,4)}
                a, b = plans[8], plans[4]
                actual = plans[den]
                check(actual.rows == local and set(actual.prefix) == pref and
                      {p-base for p in sample} == set(actual.cold or actual.rows) and
                      {p-base for p in cold} == set(actual.cold), "Counterfactual plan mismatch")
                added = sorted(set(b.rows)-set(a.rows))
                removed = sorted(set(a.rows)-set(b.rows))
                check(set(added) == set(b.prefix)-set(a.prefix) and
                      set(removed) == set(a.mask)-set(b.mask) and len(added)>=len(removed),
                      "Not a pure discretionary dose change")
                if added:
                    note.update(more_prefix=len(added), fewer_masks=len(removed),
                                extra_width=len(added)-len(removed), added=added, removed=removed,
                                state=digest([h.ordinal,h.sampler.age,sorted(init),sorted(mask),sorted(dirty)]))
            e = super()._apply(states,pos,sample,prefix,cold,offer,route)
            p, = e["parts"]
            check(tuple(e["positions"]) == tuple(pos) and e["decoder_Q"] == len(local) <= offer and
                  set(p["dirty_before"]) == dirty and set(p["prefix"]) == pref and
                  p["rows"] == list(local), "Actual call differs from recorded row classes")
            observations.append(note)
            return e
    return Dose


def first_change(a, b, tapes):
    x,y = tapes
    check(len(x)==len(a["tape"]) and len(y)==len(b["tape"]), "Missing row receipts")
    j = next((i for i,n in enumerate(y) if n["more_prefix"]), None)
    if j is None:
        check(x==y and a==b, "Unexplained change with no dose exposure")
        return None
    u,v=x[j],y[j]
    check(x[:j]==y[:j] and a["tape"][:j]==b["tape"][:j] and a["tape"][j]!=b["tape"][j]
          and all(u[k]==v[k] for k in ("state","added","removed","head","tick","dirty","route"))
          and v["prefix"]-u["prefix"]==v["more_prefix"] and u["mask"]-v["mask"]==v["fewer_masks"]
          and v["width"]-u["width"]==v["extra_width"], "Unaligned first physical change")
    return {k:v[k] for k in ("head","tick","dirty","added","removed","extra_width","state")}
