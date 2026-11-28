"""Single-request LLaDA experiment: one row planner for every positive offer."""
from bdlm_serving.methods.t2t.budget_family import Plan


def select(q, blocks, initialized, debt):
    """Blocks are ordered (lo, end, mask_positions) for live physical blocks."""
    pinned=tuple(sorted(debt)[:q])
    rows=set(pinned)
    if len(pinned)==q or not any(m for _,_,m in blocks):
        return Plan(pinned,pinned,(),(),(),True), ()
    active=[]; prefixes=[]
    for lo,end,masks in blocks:
        if masks:
            front=min(masks)
            active.extend(range(front,end))
            prefixes.extend(p for p in range(front-1,lo-1,-1) if p in initialized and p not in debt)
    room=q-len(pinned)
    # A right-edge prefix refresh can compete for at most one eighth of free seats.
    # When the active spans fit, use otherwise idle seats to update additional clean rows.
    rows.update(prefixes[:room//8])
    for p in (*active,*prefixes):
        if len(rows)==q: break
        rows.add(p)
    masks=set().union(*(m for _,_,m in blocks))
    cold=rows-initialized
    if not cold<=masks:
        raise ValueError("Invalid cold resolved row")
    live=(set(active)|set(prefixes)|{p for lo,end,m in blocks if m for p in pinned if lo<=p<end})
    sampled=tuple(sorted(rows&live)) if rows&masks else ()
    if rows and not (rows&masks) and masks-debt and room:
        # The first eligible active position is an unresolved row, by construction.
        raise ValueError("Nonempty room but no unresolved head selected")
    warm=(rows&masks&initialized)-set(pinned)
    return Plan(tuple(sorted(rows)),pinned,tuple(sorted(cold)),tuple(sorted(warm)),
                tuple(sorted(rows&set(prefixes))),not sampled),sampled


def request(base, relax_full):
    class Span(base):
        def _cross_plan(self,q):
            split=self.head+32
            lo,end=max(self.head,self.origin),min(split,self.end)
            tokens=self.x[0,lo:end].tolist()
            mask={lo+i for i,t in enumerate(tokens) if t==self.mask_id}
            opened=(not self.stopping and split<self.end and self.eos_ids.isdisjoint(tokens)
                    and (self.bank.last_end>split or (self.bank.last_end>=split and 4*len(mask)<=end-lo)))
            blocks=[] if self.stopping else [(lo,end,mask)]
            if opened:
                last=min(self.end,split+32)
                blocks.append((split,last,{split+i for i,t in enumerate(self.x[0,split:last].tolist())
                                           if t==self.mask_id}))
            p,s=select(q,blocks,self.bank.initialized,self.dirty)
            self._physical_full=bool(s) and set(range(lo,end))<=set(p.rows)
            self._cross_evidence=dict(head_masks=len(mask),opened=opened,borrowed=0)
            return p,s,"sample" if s else "drain",q

        def _additional_admissions(self,idx,old,draft,conf,accepted):
            if not relax_full or not self._physical_full:
                return accepted.new_zeros(accepted.shape)
            return (idx//32==self.head//32)[None,:] & (old==self.mask_id) & (draft!=self.mask_id) & (conf>.5)

        def _step(self,q):
            e=super()._step(q)
            e["physical_full_head"]=self._physical_full
            return e
    return Span
