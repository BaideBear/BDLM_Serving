"""Sample resident unresolved head rows already paid alongside cold MASKs."""
from .llada_window_geometry import window_class


def extra_mask(plan, sampled):
    if plan.drain_only or not plan.cold or sampled!=plan.cold:
        return ()
    return tuple(sorted(set(plan.mask)-set(plan.dirty)-set(sampled)))


def variants(base):
    control=window_class(base)

    class WindowPaidMask(control):
        def _cross_plan(self,q):
            p,s,phase,cap=super()._cross_plan(q)
            self.extra_window_mask=extra_mask(p,s) if q in (4,8,16) and self.cross=="head" and not self.borrow else ()
            return p,tuple(sorted((*s,*self.extra_window_mask))),phase,cap

        def _step(self,q):
            e=super()._step(q)
            if self.extra_window_mask:
                e["paid_waiting_mask"]=self.extra_window_mask
            return e
    return control,WindowPaidMask


def switchable_paid_class():
    """Q64 repeat request that also samples paid waiting head MASKs at Q16."""
    from .llada_window_switch import TopAgreementWindowRequest

    class SwitchablePaidWindowRequest(TopAgreementWindowRequest):
        def _switch_plan(self,q):
            plan,sampled,phase,cap=super()._switch_plan(q)
            self._paid_window_mask=()
            if (q==16 and getattr(self,"coverage","sparse")=="head"
                    and not self.borrow):
                self._paid_window_mask=extra_mask(plan,sampled)
                if self._paid_window_mask:
                    sampled=tuple(sorted((*sampled,*self._paid_window_mask)))
            return plan,sampled,phase,cap

        def _step(self,q):
            event=super()._step(q)
            paid=getattr(self,"_paid_window_mask",())
            if paid:
                event["paid_waiting_mask"]=paid
            return event

    return SwitchablePaidWindowRequest


def first(old,new):
    from .llada_fixed_family_eval import require
    for a,b in zip(old["trace"],new["trace"]):
        extra=b.get("paid_waiting_mask")
        if not extra:
            require(a==b,"different trajectory before a paid MASK was added")
            continue
        require(all(a[k]==b[k] for k in ("pre_state","rows","q","dirty","prefix","mask","cold","visible_kv",
                                          "phase","sample_clock","layer_physical_q"))
                and a["cold"] and a["cold"]==a["sampled"]
                and set(extra)==set(a["mask"])-set(a["dirty"])-set(a["sampled"])
                and set(extra)<=set(a["rows"]) and set(extra).isdisjoint(a["prefix"])
                and set(b["sampled"])==set(a["sampled"])|set(extra)
                and b["layer_physical_q"]==[b["q"]]*20,
                "changed paid work or sample clock on matched inputs")
        return dict(tick=b["tick"],width=b["q"],added=list(extra))
    require(len(old["trace"])==len(new["trace"]) and all(old[k]==new[k] for k in
            ("failure","F","N","Q","output_ids","score")),"unexplained no-exposure difference")
    return None
