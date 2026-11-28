"""One discretionary warm Q32 seat, ranked only by past unsuccessful real samples."""
from .gemma_joint_verify_eval import check, digest

B,Q=256,32
FULL=frozenset(range(B))


def choose(rows,dirty,accepted,prefix,fail):
    rows,dirty,accepted,prefix=map(set,(rows,dirty,accepted,prefix))
    masks=FULL-accepted
    check(rows<=FULL and len(rows)==Q and dirty<=rows and prefix<=rows & accepted
          and not prefix & dirty and masks and len(fail)==B, "Invalid warm sets")
    outside=masks-rows
    eligible=(rows & masks)-dirty-{min(masks)}
    if not (outside and eligible):
        return False,None
    drop=max(eligible)
    take=min(outside,key=lambda r:(fail[r],r))
    return True,((drop,take) if fail[take]<fail[drop] else None)


def advance(count,sampled,accepted_after):
    for r in sampled:
        if r not in accepted_after:
            count[r]+=1


def arm_class(base_class,enabled,notes):
    class WarmFewerFail(base_class):
        def _apply(self,states,pos,sample,prefix,cold,offer,route):
            h,=states
            r=h.rows
            check(self.fixed_q==offer==Q and self.tail is None and not self.rebuild and self.early
                  and self.law.num==1 and self.law.den==8 and self.top_gate=="off"
                  and self.fuse_first_verify and not self.reuse_paid_masks,"Wrong reference path")
            if getattr(self,"_failure_head",None)!=h.ordinal:
                check(not r.initialized and not r.age,"Not a fresh head")
                self._failure_head,self._failure_count=h.ordinal,[0]*B
            count=self._failure_count
            origin=self.origin+B*h.ordinal
            local=lambda seq:{p-origin for p in seq}
            nominal=local(pos)
            debt=set(r.dirty)
            acc=set(r.accepted)
            check(debt<=nominal<=FULL and len(pos)<=Q,"Unpaid or wrong physical plan")
            warm=bool(route=="sparse" and sample and r.initialized==FULL)
            note=dict(tick=self.tick,head=h.ordinal,warm=warm,contest=False,swap=None)
            if warm and len(pos)==Q:
                check(not cold and h.can_sample and set(sample)==set(pos) and len(h.sampler.sc)==B,
                      "Warm call scope")
                note["contest"],swap=choose(nominal,debt,acc,local(prefix),count)
                if swap:
                    drop,take=swap
                    alt=(nominal-{drop})|{take}
                    check(len(alt)==Q and debt|local(prefix)<=alt and
                          {drop,take}.isdisjoint(debt|acc|{min(FULL-acc)}),"Changed protected work")
                    note["swap"]=dict(drop=drop,take=take,counts=[count[drop],count[take]],age=r.age,
                                debt=sorted(debt),prefix=sorted(local(prefix)),ref=sorted(nominal),alt=sorted(alt),
                                state=digest([h.ordinal,r.age,sorted(acc),sorted(debt),count]))
                    if enabled:
                        pos=tuple(sorted(origin+i for i in alt))
                        sample=set(pos)
            e=super()._apply(states,pos,sample,prefix,cold,offer,route)
            p,=e["parts"]
            check(tuple(e["positions"])==tuple(pos) and set(p["rows"])==local(pos) and
                  set(p["sampled"])==local(sample) and set(p["prefix"])==local(prefix) and
                  p["age_after"]==p["age_before"]+bool(sample) and set(p["dirty_before"])==debt,
                  "Bad actual seat, clock, payment or prefix")
            advance(count,p["sampled"],r.accepted)
            notes.append(note)
            return e
    return WarmFewerFail


def align(x,y,a,b):
    check(len(a)==len(x["tape"]) and len(b)==len(y["tape"]), "Missing events")
    i=next((j for j,v in enumerate(b) if v["swap"]),None)
    if i is None:
        check(x==y and a==b,"Unexplained divergence")
        return None
    check(a[:i+1]==b[:i+1] and x["tape"][:i]==y["tape"][:i] and x["tape"][i]!=y["tape"][i],
          "Different first action or state")
    return b[i]
