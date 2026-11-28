"""One M2T factor at B and 2B: use a contiguous paid run at the warm head frontier."""


PROFILE = "quarter_window_repeat64_paid16_frontier32"


def frontier_class(control, *, solo=False, only_offer=None):
    """Keep normal and consecutive-vote admissions; add a paid left-hand run."""
    class Frontier(control):
        def _step(self, q):
            self._left = None
            lo, end = max(self.head,self.origin), min(self.head+32,self.end)
            if (only_offer is None or q==only_offer) and not self.stopping and self.bank.last_end>=end:
                pending=(self.x[0,lo:end]==self.mask_id).nonzero()
                if len(pending):
                    self._left=lo+int(pending[0,0])
            return super()._step(q)

        def _additional_admissions(self,idx,old,draft,conf,accepted):
            import torch
            inherited=getattr(super(),"_additional_admissions",None)
            extra=inherited(idx,old,draft,conf,accepted) if inherited else torch.zeros_like(accepted)
            if self._left is None or (solo and self.bank.last_end>self.head+32):
                return extra
            # A gap (not sampled), a resolved position, or a weak/MASK proposal ends the run.
            left=self._left
            valid=(idx>=left) & (idx<self.head+32) & (old[0]==self.mask_id) & (draft[0]!=self.mask_id) & (conf[0]>.5)
            coverage=torch.zeros(32,dtype=torch.bool,device=idx.device)
            coverage[idx[valid]-left]=True
            length=coverage.to(torch.int32).cumprod(0).sum()
            return extra | (valid & (idx-left<length))[None,:]
    return Frontier


def audit(trace,mask,q,enabled,*,solo=False,excluded=None):
    """Rebuild every acceptance from payments and the independently tracked token state."""
    decoded=set()
    last={}
    additions=[]
    for t,e in enumerate(trace):
        s=e["pre_state"]; head=s["head"]; lo=max(s["origin"],head); end=min(s["end"],head+32)
        warm=s["last_end"]>=end
        rows=e["sampled"]; vals=e.get("proposals",[])
        if len(vals)!=len(rows) or len(set(rows))!=len(rows):
            raise ValueError("Missing measured proposals")
        proposals=dict(zip(rows,vals)); groups={}
        for p,(old,draft,confidence,low,high) in proposals.items():
            if (old!=mask)!=(p in decoded):
                raise ValueError("Reconstructed token state disagrees with sampled input")
            if old==mask:
                groups.setdefault(p//32, []).append(p)
        natural=set()
        for positions in groups.values():
            above=[p for p in positions if proposals[p][4]]
            natural.update(above or [max(positions,key=lambda p:proposals[p][2])])
        normal={p for p in natural if proposals[p][1]!=mask}
        curr={p:proposals[p][1] for p in rows if q==64 and warm and head<=p<end and p not in natural
              and proposals[p][0]==mask and proposals[p][1]!=mask and proposals[p][3]}
        normal.update(p for p,v in curr.items() if last.get(p)==(head,t-1,v))
        last={p:(head,t,v) for p,v in curr.items()}
        run=set()
        if warm and not s["stopping"]:
            left=next((p for p in range(lo,end) if p not in decoded),end)
            for p in range(left,end):
                v=proposals.get(p)
                if not v or v[0]!=mask or v[1]==mask or not v[3]:
                    break
                run.add(p)
        if solo and (s["last_end"]>head+32 or any(p>=head+32 for p in e["rows"])):
            if excluded is not None:
                excluded.append(len(run-normal))
            run=set()
        if set(e["admitted"])!=normal|(run if enabled else set()):
            raise ValueError("Added admissions not explained by paid contiguous frontier")
        decoded.update(e["admitted"]);decoded.difference_update(e["revoked"])
        additions.append(run-normal)
    return additions


def check_first(old,new,mask,q,*,solo=False,excluded=None):
    off=audit(old["trace"],mask,q,False,solo=solo)
    on=audit(new["trace"],mask,q,True,solo=solo,excluded=excluded)
    first=None
    for i,(a,b) in enumerate(zip(old["trace"],new["trace"])):
        if not on[i]:
            if a!=b:
                raise ValueError("Trajectory changed before frontier admission")
            continue
        if (off[i]!=on[i] or any(a[k]!=b[k] for k in
                ("rows","sampled","proposals","q","dirty_before","edited","revoked",
                 "layer_physical_q","phase","sample_clock","pre_state","tail_opened"))
                or set(b["admitted"])!=set(a["admitted"])|on[i]
                or set(b["dirty_after"])!=set(a["dirty_after"])|on[i]):
            raise ValueError("First intervention was not isolated")
        first=dict(tick=i,rows=sorted(on[i]),width=a["q"])
        break
    if not first and (len(old["trace"])!=len(new["trace"]) or
            any(old[k]!=new[k] for k in ("score","N","F","Q","failure","output_ids"))):
        raise ValueError("Unexplained outcome divergence")
    return first,sum(map(len,on)),sum(map(bool,on))
