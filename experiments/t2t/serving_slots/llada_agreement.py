"""Two consecutive paid proposals on the same resident current-head MASK."""
import torch


def confirmation(head, idx, old, draft, conf, accepted, previous, mask_id):
    # The memo is one token ID per physical head address, not a logits cache.
    local = idx - head
    at = local.clamp(0, 31)
    eligible = ((local >= 0) & (local < 32) & (old[0] == mask_id)
                & (draft[0] != mask_id) & (conf[0] > .5) & ~accepted[0])
    extra = eligible & (previous[at] == draft[0])
    memo = torch.full_like(previous, -1)
    memo[at[eligible]] = draft[0, eligible]
    return extra[None], memo


def request_class(control, *, fixed_q=64):
    if fixed_q not in (8,16,64):
        raise ValueError("No fixed-window repeat control at this offer")
    class Agreement(control):
        def _step(self, q):
            if q!=fixed_q or (q==64 and (self.cross!="quarter" or not self.warm_head_sample)) or (
                    q!=64 and (self.cross!="head" or self.borrow or self.warm_head_sample)):
                raise ValueError("Mismatch with registered fixed-offer control")
            self._warm_now = self.bank.last_end >= min(self.head+32,self.end)
            return super()._step(q)

        def _additional_admissions(self, idx, old, draft, conf, accepted):
            if not self._warm_now:
                return torch.zeros_like(accepted)
            tick = len(self.records)
            prev = (self._agree_memo if getattr(self,"_agree_stamp",None)==(self.head,tick-1)
                    else torch.full((32,),-1,device=idx.device,dtype=draft.dtype))
            extra, self._agree_memo = confirmation(self.head,idx,old,draft,conf,accepted,prev,self.mask_id)
            self._agree_stamp = (self.head,tick)
            return extra
    return Agreement


def inspect_trace(trace, mask, enabled):
    """Reconstruct admissions independently from paid proposals and known input state."""
    memo, events, paid = {}, [], 0
    for t,e in enumerate(trace):
        pre=e["pre_state"]
        head=pre["head"]
        warm=pre["last_end"]>=min(head+32,pre["end"])
        if len(e["sampled"])!=len(e.get("proposals",[])):
            raise ValueError("Missing sampled evidence")
        natural, groups={},{}
        for p, (old,draft,conf,low,high) in zip(e["sampled"],e.get("proposals",[])):
            if old!=mask:
                continue
            groups.setdefault(p//32,[]).append((p,conf))
            natural[p]=bool(high)
        for group in groups.values():
            if not any(natural[p] for p,_ in group):
                natural[max(group,key=lambda x:x[1])[0]]=True
        extra, incumbent, next_memo=set(),set(),{}
        for p,(old,draft,conf,low,high) in zip(e["sampled"],e.get("proposals",[])):
            if old!=mask or draft==mask:
                continue
            if natural[p]:
                incumbent.add(p)
            elif warm and head<=p<head+32 and low:
                paid+=1
                if memo.get(p)==(head,t-1,draft):
                    extra.add(p)
                next_memo[p]=(head,t,draft)
        if set(e["admitted"]) != incumbent | (extra if enabled else set()):
            raise ValueError("Paid proposals do not explain M2T")
        memo=next_memo
        events.append(sorted(extra))
    return events, paid


def first_change(old,new,mask):
    from .llada_fixed_family_eval import require
    control, _=inspect_trace(old["trace"],mask,False)
    trial, eligible=inspect_trace(new["trace"],mask,True)
    first=None
    for i,(a,b) in enumerate(zip(old["trace"],new["trace"])):
        if not trial[i]:
            require(a==b,"request changed before a second paid proposal")
            continue
        added=set(trial[i])
        require(control[i]==trial[i] and all(a[k]==b[k] for k in
                ("pre_state","rows","q","layer_physical_q","sampled","proposals","sample_clock",
                 "dirty_before","edited","revoked","tail_opened","phase"))
                and a.get("warm_head_rows")==b.get("warm_head_rows")
                and a["layer_physical_q"]==[a["q"]]*20 and not (added & set(a["admitted"]))
                and set(b["admitted"])==set(a["admitted"])|added
                and set(b["dirty_after"])==set(a["dirty_after"])|added,
                "First difference not isolated to eligible head M2T")
        first=dict(tick=i,width=a["q"],added=sorted(added))
        break
    if first is None:
        require(len(old["trace"])==len(new["trace"]) and all(old[k]==new[k] for k in
                ("output_ids","F","Q","failure","score")),"unexplained difference")
    return first,dict(extra_rows=sum(map(len,trial)),extra_F=sum(map(bool,trial)),eligible=eligible)
