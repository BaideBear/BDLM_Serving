"""Physically pay all incoming writes; optionally narrow a warm sampler action."""
from hashlib import sha256
from statistics import mean

from .gemma_joint_verify_eval import check, digest

FULL=frozenset(range(256))
ANCHORS=(0,10,20,30,50,70)


def eligible(route, initialized, positions, sampled, dirty, accepted, prefix):
    check(dirty <= positions <= FULL and sampled <= positions and prefix <= positions
          and not prefix & dirty, "Unpaid or overlapping physical plan")
    if route!="sparse" or initialized!=FULL or not sampled-accepted:
        return set()
    check(sampled==positions and prefix<=accepted, "Unexpected warm plan")
    return dirty & accepted & sampled


def arm_class(base_class, enabled, tape):
    class PaymentOnly(base_class):
        def _project(self, states, pos):
            logits=super()._project(states,pos)
            if self._tap:
                import torch
                self._first_logits=sha256(logits.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            return logits

        def _apply(self,states,pos,sample,prefix,cold,offer,route):
            h,=states
            check(self.fixed_q==offer==32 and self.tail is None and not self.rebuild and self.early and
                  self.fuse_first_verify and self.top_gate=="off" and not self.reuse_paid_masks and
                  (self.law.num,self.law.den)==(1,8),"Wrong fixed control")
            base=self.origin+256*h.ordinal
            local=lambda xs:{p-base for p in xs}
            r,s=h.rows,h.sampler
            initial,debt,acc=set(r.initialized),set(r.dirty),set(r.accepted)
            nom=local(sample)
            prefix_rows=local(prefix)
            physical=local(pos)
            skip=eligible(route,initial,physical,nom,debt,acc,prefix_rows)
            self._tap=bool(skip) and not getattr(self,"_first_logits",None)
            first=self._tap
            common=None
            if skip:
                check(not cold and len(s.sc)==256 and h.can_sample and nom-acc and len(pos)<=offer,
                      "Not a warm mixed sample")
                if first:
                    common=dict(tick=self.tick,head=h.ordinal,age=s.age,pos=sorted(physical),nom=sorted(nom),
                                debt=sorted(debt),skip=sorted(skip),prefix=sorted(prefix_rows),
                                accepted_sha=digest(sorted(acc)))
            actual=nom-skip if enabled else nom
            protected=sorted(skip) if enabled else []
            if protected:
                oldids=s.ids[0,protected].clone()
                oldsc=[s.sc[i] for i in protected]
            check(bool(actual)==bool(nom) and prefix_rows & nom <= actual, "Lost a clock or prefix")
            phase=("verify" if route in ("verify","verify_paid") else "cold" if initial!=FULL
                   else "warm" if h.can_sample else "close")
            e=super()._apply(states,pos,{i+base for i in actual},prefix,cold,offer,route)
            p,=e["parts"]
            writes=set(p["input_changed"])|set(p["admitted"])|set(p["revoked"])
            check(e["positions"]==list(pos) and set(p["rows"])==physical and set(p["sampled"])==actual
                  and set(p["prefix"])==prefix_rows and set(p["dirty_before"])==debt and
                  writes<=actual and set(r.dirty)==writes and p["age_after"]==p["age_before"]+bool(nom),
                  "Changed physical schedule, writeback or clock")
            if protected:
                check(s.ids[0,protected].equal(oldids) and skip<=r.accepted and
                      all(s.sc[i] is prev for i,prev in zip(protected,oldsc)),
                      "Payment-only row mutated without sample")
            if first:
                common["logits"]=self._first_logits
            tape.append(dict(tick=e["tick"],head=h.ordinal,phase=phase,q=len(pos),first=common,
                         nominal=len(nom),sampled=len(actual),eligible=len(skip),paid=len(debt),writes=len(writes),
                         suppressed=len(protected),admit=len(p["admitted"]),revoke=len(p["revoked"]),
                         edit=len(p["edited"]),eligible_edits=len(skip & set(p["edited"])),
                         eligible_revokes=len(skip & set(p["revoked"]))))
            return e
    return PaymentOnly


def audit(x,t,enabled):
    check(len(t)==x["decoder_F"]==len(x["tape"]) and sum(v["q"] for v in t)==x["decoder_Q"] and
          all(0<v["q"]<=32 and v["tick"]==i for i,v in enumerate(t)) and
          x["F"]==len(t)+x["encoder_F"] and x["encoder_Q"]==256*x["encoder_F"],"Invalid physical bill")
    d={k:sum(v[k] for v in t) for k in ("nominal","sampled","eligible","suppressed","paid","writes",
                                         "admit","revoke","edit","eligible_edits","eligible_revokes")}
    d.update(peak=max(v["q"] for v in t),opportunity_ticks=sum(v["eligible"]>0 for v in t),
             heads=x["encoder_F"]+1)
    for phase in ("cold","warm","close","verify"):
        part=[v for v in t if v["phase"]==phase]
        d[phase+"_F"],d[phase+"_Q"]=len(part),sum(v["q"] for v in part)
    d["first_head_warm_F"]=sum(v["phase"]=="warm" and v["head"]==0 for v in t)
    check(sum(d[k+"_F"] for k in ("cold","warm","close","verify"))==len(t)
          and d["writes"]==d["paid"] and d["suppressed"]==(d["eligible"] if enabled else 0)
          and d["nominal"]-d["sampled"]==d["suppressed"],"Unbalanced sampler/debt ledger")
    return d


def align(x,y,a,b):
    i=next((j for j,v in enumerate(b) if v["first"]),None)
    if i is None:
        check(x==y and a==b, "Unexplained unexposed difference")
        return None
    u=a[i]["first"]
    check(a[:i]==b[:i] and x["tape"][:i]==y["tape"][:i] and x["tape"][i]!=y["tape"][i]
          and u==b[i]["first"] and u["skip"] and len(u["logits"])==64 and
          a[i]["sampled"]-b[i]["sampled"]==len(u["skip"]) and a[i]["q"]==b[i]["q"],
          "First divergence was not the registered change to fresh sampled rows")
    return u


def gates(rows):
    check([r["index"] for r in rows]==list(ANCHORS),"Wrong six anchors")
    return dict(old_exposure=any(r["first"] for r in rows[:3]),new_exposure=any(r["first"] for r in rows[3:]),
                quality=sum(int(r["on"]["correct"])-int(r["off"]["correct"]) for r in rows)>=0,
                macro_TPF=mean(r["on"]["N"]/r["on"]["F"]-r["off"]["N"]/r["off"]["F"] for r in rows)>0,
                lower_cost=any(sum(r["on"][k]-r["off"][k] for r in rows)<0 for k in ("F","decoder_Q")))
