"""Move only the leading cold tile to the end of first visits on a solo Gemma head."""
from .gemma_joint_verify_eval import check

B = 256
TIERS = (32, 64)
PHASES = ("cold", "warm", "closing_drain", "verify")


def first_visits(seen, dirty, q, enabled):
    check(q in TIERS and type(enabled) is bool and set(dirty)<=set(seen)<=set(range(B))
          and len(dirty)<=q, "Invalid cold schedule")
    missing=set(range(B))-set(seen)
    ranked=sorted(missing,key=lambda i:(enabled and i<q,i))
    return tuple(sorted(ranked[:q-len(dirty)]))


def arm_class(base_class, enabled, tape):
    class ColdFrontierLast(base_class):
        def _apply(self,states,pos,sample,prefix,cold,offer,route):
            h,=states
            q=offer
            check(self.fixed_q==q and q in TIERS and self.tail is None and not self.rebuild and
                  self.early and self.top_gate=="off" and self.fuse_first_verify and
                  self.law.num==1 and self.law.den==8, "Not the registered sparse EOS profile")
            base=self.origin+B*h.ordinal
            r=h.rows
            seen,dirty,accepted=set(r.initialized),set(r.dirty),set(r.accepted)
            if len(seen)<B:
                old=first_visits(seen,dirty,q,False)
                check(set(sample)==set(cold)=={base+i for i in old} and not prefix and
                      set(pos)==set(sample)|{base+i for i in dirty} and
                      route==("sparse" if old else "drain"),"Wrong baseline cold plan")
                first=first_visits(seen,dirty,q,enabled)
                cold=tuple(base+i for i in first)
                sample=set(cold)
                pos=tuple(sorted(sample|{base+i for i in dirty}))
                phase="cold"
            elif route in ("verify","verify_paid"):
                phase="verify"
            else:
                phase="warm" if h.can_sample else "closing_drain"
            local=tuple(p-base for p in pos)
            check(dirty<=set(local) and 0<len(local)<=q,"Unpaid or over budget")
            e=super()._apply(states,pos,sample,prefix,cold,q,route)
            p,=e["parts"]
            check(tuple(e["positions"])==pos and p["rows"]==list(local) and
                  set(p["sampled"])=={i-base for i in sample} and set(p["dirty_before"])==dirty and
                  p["age_after"]==p["age_before"]+bool(sample),"Physical plan or clock drift")
            tape.append(dict(tick=e["tick"],head=h.ordinal,phase=phase,route=route,width=len(pos),
                             cold=p["sampled"] if phase=="cold" else [],sample=len(sample),
                             dirty=len(dirty),prefix=len(p["prefix"]),age=p["age_before"],
                             unresolved=B-len(accepted),first_tile_masks=q-len(set(range(q))&accepted),
                             writes=len(set(p["admitted"])|set(p["revoked"])|set(p["input_changed"]))))
            return e
    return ColdFrontierLast


def audit(x,tape,q,enabled):
    check(len(tape)==len(x["tape"])==x["decoder_F"] and
          x["F"]==x["decoder_F"]+x["encoder_F"] and x["encoder_Q"]==B*x["encoder_F"],
          "Bad physical totals")
    by_head={}
    for j,e in enumerate(tape):
        check(e["tick"]==j and 0<e["width"]<=q and 0<=e["sample"]<=e["width"] and
              0<=e["writes"]<=e["sample"] and e["dirty"]<=e["width"],"Bad tick")
        by_head.setdefault(e["head"],[]).append(e)
    check(set(by_head)==set(range(1+x["encoder_F"])),"Missing heads")
    cold_ticks=warm_entry_frontier_masks=0
    for steps in by_head.values():
        seen=set()
        order=[]
        age=0
        after_cold=False
        for e in steps:
            check(e["age"]==age,"Clock discontinuity")
            age+=bool(e["sample"])
            if e["phase"]=="cold":
                chunk=e["cold"]
                check(len(seen)<B and not after_cold and not seen.intersection(chunk)
                      and len(chunk)==e["sample"] and len(chunk)+e["dirty"]==e["width"]
                      and len(chunk)==min(B-len(seen),q-e["dirty"]),"Starved or mixed cold call")
                seen.update(chunk)
                order.extend(chunk)
                cold_ticks+=bool(chunk)
            elif not after_cold:
                check(len(seen)==B,"Entered warm/close before coverage")
                warm_entry_frontier_masks+=e["first_tile_masks"]
                after_cold=True
        expected=list(range(q,B))+list(range(q)) if enabled else list(range(B))
        # A single physical batch must be sorted even if it straddles the wrap.
        k=0
        for e in steps:
            n=len(e["cold"])
            check(e["cold"]==sorted(expected[k:k+n]),"Not the registered cyclic order")
            k+=n
        check(k==len(order)==B and seen==set(range(B)) and after_cold and steps[-1]["phase"]=="verify",
              "Incomplete head")
    result=dict(heads=len(by_head),cold_sample_F=cold_ticks,peak=max(e["width"] for e in tape),
                warm_entry_frontier_masks=warm_entry_frontier_masks,
                cold_Q_first=B*len(by_head),dirty_Q=sum(e["dirty"] for e in tape),
                prefix_Q=sum(e["prefix"] for e in tape),writes=sum(e["writes"] for e in tape))
    for phase in PHASES:
        part=[e for e in tape if e["phase"]==phase]
        result.update({phase+"_F":len(part),phase+"_Q":sum(e["width"] for e in part)})
    check(sum(result[p+"_F"] for p in PHASES)==x["decoder_F"] and
          sum(result[p+"_Q"] for p in PHASES)==x["decoder_Q"],"Unreconciled phase bill")
    return result
