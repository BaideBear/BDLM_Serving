"""Spend the existing non-prefix, non-dirty seats on the first live span."""
from dataclasses import replace


def window(plan, sampled, masks, initialized, end):
    """IBS-style interior candidates but retain this planner's prefix dose and width."""
    pinned=set(plan.prefix)|set(plan.dirty)
    slots=len(plan.rows)-len(pinned)
    if not sampled or not slots or not masks:
        return plan,sampled,None
    interior=(set(range(min(masks),end)) & initialized)-masks-pinned
    free_mask=sorted(masks-pinned)
    candidates=interior|set(free_mask[:max(1,slots-len(interior))])
    picked=set(sorted(candidates)[:slots])
    if len(picked)!=slots:
        raise ValueError("Cannot spend the originally measured width")
    rows=tuple(sorted(pinned|picked))
    if rows==plan.rows:
        return plan,sampled,None
    cold=tuple(sorted(picked-initialized))
    warm=tuple(sorted(picked&masks&initialized))
    if any(p not in masks for p in cold):
        raise ValueError("Uninitialized decoded interior")
    # Identical adapter rule: only cold proposals on a cold call; otherwise
    # every paid active-head row can sample. Do not synthesize old logits.
    sample=cold or rows
    p=replace(plan,rows=rows,cold=cold,mask=warm)
    return p,sample,dict(before=plan.rows,interior=tuple(sorted(picked&interior)))


def window_class(control):
    class Window(control):
        def _cross_plan(self,q):
            p,s,phase,cap=super()._cross_plan(q)
            self.geometry=None
            if self.cross=="head" and not self.borrow and q in (8,16) and s:
                masks,_,_=self._inputs(32)
                p,s,self.geometry=window(p,s,masks,self.bank.initialized,min(self.head+32,self.end))
            return p,s,phase,cap

        def _step(self,q):
            event=super()._step(q)
            if self.geometry:
                event["window_geometry"]=self.geometry
            return event
    return Window


def first_geometry(old,new):
    from .llada_fixed_family_eval import require
    for x,y in zip(old["trace"],new["trace"]):
        g=y.get("window_geometry")
        if g is None:
            require(x==y,"unexplained change before first window intervention")
            continue
        before,after=map(set,(x["rows"],y["rows"]))
        pinned=set(x["dirty"])|set(x["prefix"])
        require(tuple(x["rows"])==g["before"] and before!=after and pinned<=after and
                all(x[k]==y[k] for k in ("pre_state","dirty","prefix","q","layer_physical_q","phase")) and
                x["layer_physical_q"]==[len(after)]*20 and len(after)==len(before) and
                y["sampled"]==(y["cold"] or y["rows"]) and
                set(g["interior"]).isdisjoint(pinned|set(y["cold"])|set(y["mask"])),
                "unmatched first window change")
        return dict(tick=x["tick"],width=x["q"],added=sorted(after-before),removed=sorted(before-after),
                    interior=list(g["interior"]),old_visible=x["visible_kv"],new_visible=y["visible_kv"])
    require(len(old["trace"])==len(new["trace"]) and all(old[k]==new[k] for k in
            ("failure","F","N","Q","output_ids","score")),"no intervention but results differ")
    return None
