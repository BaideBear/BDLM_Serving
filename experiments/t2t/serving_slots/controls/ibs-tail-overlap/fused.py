"""Qualified single-forward overlap on the frozen B32 fixed-IBS runner."""
from hashlib import sha256
from inspect import getsource
from pathlib import Path
from textwrap import dedent

import torch

from bdlm_serving.methods.t2t.close_overlap import may_promote_successor, spare_successor_rows


def checked(ok, why):
    if not ok:
        raise ValueError(why)


class Successor:
    def __init__(self, apply, receipt):
        self.apply, self.receipt = apply, receipt
        self.tried = set()

    def plan(self, x, p, cap, start, offer, head, old_mask, dirty, cache, eos):
        checked(0 < len(head) <= offer <= 32, "invalid fixed IBS offer")
        if len(head) == offer or start in self.tried or not old_mask.all() or start+32 >= p+cap:
            return ()
        head_set = set(range(start,start+32))
        # Only generated IDs in the current head can veto a successor.
        has_eos = bool(set(x[0,max(p,start):min(start+32,p+cap)].tolist()) & set(eos))
        rows = spare_successor_rows(offer=offer,head_start=start,width=32,generated_end=p+cap,
                                    selected_head=head.tolist(),dirty_head=dirty,
                                    resident_head=head_set & cache.initialized,unresolved_head=(),
                                    generated_eos=has_eos,already_tried=False)
        if rows:
            self.tried.add(start)
            checked(not set(rows) & cache.initialized, "unexpected resident successor")
            self.receipt.append(dict(head_start=start,head_rows=head.tolist(),tail_rows=list(rows)))
        return rows if self.apply else ()


def joint_mask(positions, nhead, end):
    # IBS head's old right edge, not the end of its whole physical block.
    keys = torch.arange(end,device=positions.device)
    ends = torch.where(torch.arange(len(positions),device=positions.device) < nhead,
                       positions[nhead-1]+1, positions[-1]+1)
    return keys[None,:] < ends[:,None]


def dual_sample(engine, tail_logits, start, tail_abs):
    ids = tail_logits[0].argmax(dim=-1)
    mask_state = torch.zeros(32,device=ids.device,dtype=torch.uint8)
    loc = tail_abs-(start+32)
    updated = engine.unmask.low_confidence_dynamic_sparse(tail_logits[0],ids,mask_state,loc,row_width=32)
    checked(not bool(updated[len(loc):].any()), "sampler accepted an unmeasured tail row")
    return ids, updated[loc].bool()


def settle(event, cache, start, ready, changed, x, dirty, head, tail, ids, take, mask_id,
           p, gen_length, eos):
    checked(not bool((x[0,tail]!=mask_id).any()), "tentative row was not an untouched MASK")
    selected = take & (ids!=mask_id)
    proposed_writes = tail[selected].tolist()
    ended = (start+32>=p+gen_length or
             bool(set(x[0,max(p,start):min(start+32,p+gen_length)].tolist()) & set(eos)))
    # LLaDA has no encoder update at promotion. Any head edit or nonclosure
    # invalidates this speculation, regardless of agreement of output IDs.
    promoted = may_promote_successor(head_sealed=ready==32 and not dirty,head_changed=changed,
                                     request_ended=ended,projection_context=start,
                                     promotion_context=start)
    head_end, tail_end = int(head[-1])+1,int(tail[-1])+1
    if promoted:
        x[0,tail[selected]] = ids[selected]
        dirty.update(proposed_writes)
        event["written_absolute"] = sorted(set(event["written_absolute"])|set(proposed_writes))
    else:
        # Values can remain allocated, but may never be read before overwrite.
        cache.initialized.difference_update(tail.tolist())
        cache.selected = head
        cache.last_end = cache.visible_end = head_end
    event.update(cross_boundary_rows=len(tail), successor=dict(
        head_rows=head.tolist(),tail_rows=tail.tolist(),tail_proposed=ids.tolist(),
        tail_writes=proposed_writes if promoted else [],
        discarded_writes=[] if promoted else proposed_writes,
        head_changed=changed,promoted=promoted,head_key_end=head_end,tail_key_end=tail_end,
        layer_widths=list(cache.joint_widths)))


def install_run(fixed, *, arm, receipt):
    """Inject four guarded seams into the pinned runner; stock does not use this function."""
    checked(arm in ("noop","overlap"),"unexpected arm")
    checked(sha256(Path(fixed.__file__).read_bytes()).hexdigest()==
            "b95003b5f8867bd6ef7d5126c9814e08fb6d03bab57f67642618028d8a1960d4","unexpected stock source")
    class WidthCache(fixed.SelectedResidentCache):
        joint_n = 0
        def update(self,key,value,layer_idx,cache_kwargs=None):
            if self.joint_n:
                checked(layer_idx==len(self.joint_widths) and
                        key.shape[-2]==value.shape[-2]==len(self.selected)==self.joint_n,
                        "unexpected physical layer width")
                self.joint_widths.append(self.joint_n)
            return super().update(key,value,layer_idx,cache_kwargs)
    src = dedent(getsource(fixed.FixedOptimus.run))
    def hook(old,new):
        nonlocal src
        checked(src.count(old)==1,"ambiguous stock source hook: "+old)
        src=src.replace(old,new)
    hook("    events = []","    events = []\n    spec = Successor(apply,receipt)\n"
         "    checked(self.kind=='llada21' and self.mode=='ibs' and self.caps==(8,) and not self.prefix_refresh, 'unsupported study')")
    hook("def forward(positions, phase):","def forward(positions, phase, nhead=None):")
    hook('        if phase == "prefill":','        if nhead is not None:\n'
         '            visible = joint_mask(positions,nhead,cache.visible_end)\n'
         '        elif phase == "prefill":')
    hook("            logits = forward(projected, \"optimus\")",'''            tail = spec.plan(x,p,gen_length,start,offer,positions,old_mask,dirty,cache,eos_ids)
            if tail:
                tail = torch.as_tensor(tail,device=x.device,dtype=positions.dtype)
                fused = torch.cat([positions,tail])
                checked(len(fused)<=offer, "over budget")
                cache.joint_n, cache.joint_widths = len(fused), []
                all_logits = forward(fused,"optimus",nhead=len(positions))
                cache.joint_n = 0
                checked(len(cache.joint_widths)==len(cache.buffers)==self.model.config.num_hidden_layers,
                        "not every decoder layer projected the joint packet")
                logits = all_logits[:,:len(positions)]
                tail_ids, tail_take = dual_sample(self,all_logits[:,len(positions):],start,tail)
                del all_logits
            else:
                logits = forward(projected, "optimus")''')
    hook("            assert not bool((positions[edited] < start+ready).any())",
         '''            if len(tail):
                settle(events[-1],cache,start,ready,bool(edited.any()),x,dirty,positions,tail,
                       tail_ids,tail_take,mask_id,p,gen_length,eos_ids)
            assert not bool((positions[edited] < start+ready).any())''')
    ns = dict(vars(fixed), Successor=Successor, checked=checked, joint_mask=joint_mask, settle=settle,
              dual_sample=dual_sample,apply=arm=="overlap",receipt=receipt,SelectedResidentCache=WidthCache)
    exec(compile(src,"<pinned-fixed-ibs-joint-study>","exec"),ns)
    return ns["run"]
