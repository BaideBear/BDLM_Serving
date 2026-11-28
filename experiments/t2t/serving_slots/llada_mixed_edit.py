"""Allow native editing of clean, already projected interior rows on mixed cold Q16 calls."""
from .llada_fixed_family_eval import require


def extra(plan, sample, masks, resident):
    if not plan.cold or plan.drain_only or not sample:
        return ()
    return tuple(sorted((set(plan.rows) & resident)-set(plan.dirty)-set(plan.prefix)
                        -masks-set(sample)))


def variant(paid):
    class Mixed(paid):
        def _cross_plan(self,q):
            p,s,phase,cap=super()._cross_plan(q)
            self.mixed_edit=()
            if q==16 and self.cross=="head" and not self.borrow and p.cold:
                masks,_,_=self._inputs(32)
                self.mixed_edit=extra(p,s,masks,self.bank.initialized)
                require(all(min(masks)<r<self.head+32 for r in self.mixed_edit),"not decoded interior")
            return p,tuple(sorted((*s,*self.mixed_edit))),phase,cap

        def _step(self,q):
            e=super()._step(q)
            if self.mixed_edit:
                e["mixed_edit"]=self.mixed_edit
            return e
    return Mixed


def inspect(a,b,mask_id):
    decoded=set()
    first=None
    counts=dict(calls=0,seats=0,T2T=0,revoked=0)
    changed=False
    for i,y in enumerate(b["trace"]):
        added=set(y.get("mixed_edit",()))
        if added:
            st=y["pre_state"]; lo=max(st["head"],st["origin"]); hi=min(st["head"]+32,st["end"])
            frontier=next(p for p in range(lo,hi) if p not in decoded)
            require(y["cold"] and all(frontier<p<min(st["last_end"],hi) for p in added)
                    and added==(set(y["rows"]) & decoded)-set(y["dirty"])-set(y["prefix"])
                    and added<=set(y["sampled"]) and all(y["proposals"][y["sampled"].index(p)][0]!=mask_id
                                                         for p in added),"not measured clean decoded rows")
            counts["calls"]+=1;counts["seats"]+=len(added)
            counts["T2T"]+=len(added & set(y["edited"]))
            counts["revoked"]+=len(added & set(y["revoked"]))
        if not changed:
            require(i<len(a["trace"]),"unexplained additional call")
            x=a["trace"][i]
            if added:
                require(all(x[k]==y[k] for k in ("pre_state","q","rows","dirty","prefix","cold",
                        "mask","layer_physical_q","phase","sample_clock","visible_kv"))
                        and set(y["sampled"])-set(x["sampled"])==added
                        and set(x["sampled"])<=set(y["sampled"]) and x["admitted"]==y["admitted"]
                        and all(x["proposals"][j]==y["proposals"][y["sampled"].index(p)]
                                for j,p in enumerate(x["sampled"]))
                        and all(set(x[k])==set(y[k])-added for k in ("edited","revoked","written_absolute")),
                        "unmatched work or multiple causal factors")
                if first is None: first=dict(tick=i,width=y["q"],added=sorted(added))
                changed=bool(added & set(y["written_absolute"]))
            else:
                require(x==y,"different without edit")
        decoded.update(y["admitted"]);decoded.difference_update(y["revoked"])
    if not changed:
        require(len(a["trace"])==len(b["trace"]) and all(a[k]==b[k] for k in
                ("output_ids","score","N","F","Q","failure")),"unexplained trajectory difference")
    return first,counts
