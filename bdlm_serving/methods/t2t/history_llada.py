"""Route only explicitly scoped LLaDA denoise calls through split attention."""
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, field

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .history_attention import SlotPlan, block_history, state_like


def _bounds(mask):
    if mask.shape[-1] == 0:
        return None
    allowed = mask if mask.dtype == torch.bool else mask == 0
    width = allowed.shape[-1]
    row = torch.arange(width,device=mask.device)
    lo = torch.where(allowed,row,width).amin(-1)
    hi = torch.where(allowed,row+1,0).amax(-1)
    if not torch.equal(allowed.sum(-1), (hi-lo).clamp_min(0)):
        raise ValueError("non-contiguous legal KV set; cannot represent by a range")
    if bool(((lo==0)&(hi==width)).all()):
        return None
    return torch.stack((lo.minimum(hi),hi),-1).int().contiguous()


@dataclass
class _Scope:
    selected: list
    states: dict
    plan: object = None
    bounds: tuple = ()
    seen: set = field(default_factory=set)


class LLadaHistory:
    def __init__(self,model):
        self.layers=[layer.attention for layer in model.model.layers]
        self.active=None
        self.saved=[]
        self.key=f"bdlm_llada_hist_{id(self):x}"
        if any(x.config._attn_implementation!="sdpa" for x in self.layers):
            raise ValueError("only pinned SDPA reference supported")
        self.native=ALL_ATTENTION_FUNCTIONS["sdpa"]
        ALL_ATTENTION_FUNCTIONS[self.key]=self._call
        for x in self.layers:
            self.saved.append(x.config)
            x.config=copy(x.config)
            x.config._attn_implementation=self.key

    def close(self):
        for x,c in zip(self.layers,self.saved):
            x.config=c
        del ALL_ATTENTION_FUNCTIONS[self.key]

    @contextmanager
    def activate(self, selected, cache_states=None):
        if self.active is not None:
            raise RuntimeError("overlapping forwards")
        scope=_Scope(selected,{} if cache_states is None else cache_states)
        self.active=scope
        try:
            yield scope
            if len(scope.seen)!=len(self.layers):
                raise RuntimeError("forward did not route every decoder layer")
        finally:
            self.active=None

    def _call(self,module,q,k,v,mask,**kw):
        scope=self.active
        if scope is None:
            return self.native(module,q,k,v,mask,**kw)
        r,_,b,_=q.shape
        h=k.shape[-2]-b
        if b!=32 or h<0 or kw.get("dropout",0):
            raise ValueError("expected one complete native block in eval mode")
        i=module.layer_idx
        if i in scope.seen:
            raise RuntimeError("second forward in a single selection scope")
        scope.seen.add(i)
        if scope.plan is None:
            scope.plan=SlotPlan.make(scope.selected,b,q.device)
            if mask is None:
                scope.bounds=(None,None)
            else:
                if mask.ndim!=4 or mask.shape[1]!=1 or mask.shape[-1]!=h+b:
                    raise ValueError("unexpected native attention mask")
                legal=mask[:,0].expand(r,b,h+b)
                scope.bounds=(_bounds(legal[...,h:]),_bounds(legal[...,:h]))
        if i not in scope.states:
            scope.states[i]=state_like(q)
        out=block_history(q.contiguous(),k[:,:,h:].contiguous(),v[:,:,h:].contiguous(),
                          k[:,:,:h].contiguous(),v[:,:,:h].contiguous(),
                          *scope.bounds,scope.plan,scope.states[i],kw["scaling"])
        return out.transpose(1,2).contiguous(),None
