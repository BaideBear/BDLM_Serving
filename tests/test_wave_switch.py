import copy
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from bdlm_serving.methods.t2t.wave_offer import audit,periodic,widths
from bdlm_serving.runtime.diffusiongemma_wave import generate


class Model:
    """A physical decoder/encoder tape; each canvas admits, edits, then stops."""
    def __init__(self,limit=5,early_tail=False,context_sensitive=False):
        self.config=NS(canvas_length=4,text_config=NS(layer_types=["full","sliding"]))
        self.cfg=NS(max_denoising_steps=limit,eos_token_id=[9])
        self.model=NS(encoder=self.encode,decoder=NS(self_conditioning=nn.Identity()))
        self.events=[]
        self.steps=[]
        self.early_tail=early_tail
        self.context_sensitive=context_sensitive
    def _prepare_generation_config(self,*args,**kwargs):
        return self.cfg,{}
    def _prepare_cache_for_generation(self,*args):
        layer=lambda:NS(keys=torch.empty(1,1,0,1))
        return NS(layers=[layer(),layer()],is_compileable=False)
    def _prepare_sampler(self,cfg):
        return NS(initialize_canvas=lambda n,d:torch.zeros(1,4,device=d,dtype=torch.long),
                  accepted_token_mask=torch.zeros(1,4,dtype=torch.bool))
    def _prepare_diffusion_stopping_criteria(self,cfg):
        return object()
    def _prepare_logits_processor(self,cfg):
        return None
    def _prepare_encoder_inputs(self,ids,mask,pos,cache,*args,**kw):
        return ids[:,-pos.numel():],None
    def encode(self,input_ids,past_key_values,position_ids,**_):
        self.events.append(("encoder",position_ids.numel()))
        for layer in past_key_values.layers:
            layer.keys=torch.empty(1,1,1+int(position_ids.max()),1)
        return NS(past_key_values=past_key_values)
    def __call__(self,decoder_input_ids,decoder_position_ids,decoder_attention_mask,past_key_values,**_):
        q=decoder_input_ids.numel()
        self.events.append(("decoder",q))
        if q==8:
            for mask in decoder_attention_mask.values():
                assert not mask[0,0,:4,-4:].any() and mask[0,0,4:,-8:].all()
        logits=torch.zeros(1,q,10)
        logits[:,:,0]=decoder_position_ids
        return NS(logits=logits)
    def _denoising_step(self,decoder_forward,current_canvas,sampler,cur_step,input_ids,**_):
        raw=decoder_forward().logits
        start=int(raw[0,0,0])
        age=self.cfg.max_denoising_steps-cur_step
        assert 0<=age<self.cfg.max_denoising_steps
        self.steps.append((start,input_ids.numel(),age))
        ids=current_canvas.clone()
        ids[0,:min(age+1,4)]=2+age%2
        if self.context_sensitive and start>2:
            ids[0,0]=7 if input_ids.numel()==start else 5
        sampler.accepted_token_mask[0,:min(age+1,4)]=True
        stop=age>=3 or (self.early_tail and start>input_ids.numel() and age>=1)
        return ids,ids.clone(),raw,stop


def run(policy,offers=None,guard=True,**options):
    m=Model(**options)
    series=periodic(offers) if offers else None
    out,tr=generate(m,dict(input_ids=torch.tensor([[6,7]]),attention_mask=torch.ones(1,2)),
        policy,dict(block_size=4,cap=16,flowblock_spawn_threshold=0),next_budget=series,
        verify_promotion=guard)
    if series:
        audit(tr,m.events,4,offers)
    return out,tr,m


@pytest.mark.parametrize("q,ref",[(4,"flow_w1"),(8,"active_w2")])
def test_constant_offers_reproduce_projection_trajectory(q,ref):
    a,x,m=run(ref)
    b,y,n=run("external_w2",(q,),guard=False)
    assert a==b and m.events==n.events and m.steps==n.steps
    assert x==[{k:s[k] for k in x[0]} for s in y]


def test_park_after_admission_and_strict_edit_then_promote_once():
    out,tr,m=run("external_w2",(8,8,8,4,4,4),early_tail=True)
    flush=tr[3]
    assert tr[2]["strict_edits"][1] and tr[2]["admitted"][1]
    assert flush["offer"]==4 and flush["q"]==8 and flush["refresh_only"]==1
    assert flush["ages_before"][1]==flush["ages_after"][1] and flush["dirty_after"][1]==[]
    assert flush["promoted"]==[1] and tr[4]["head"]==1 and tr[4]["q"]==4
    assert tr[4]["ages_after"][0]==tr[4]["ages_before"][0]+1
    assert (6,6,2) in m.steps
    bad=copy.deepcopy(tr)
    bad[3]["q"]=4
    with pytest.raises(ValueError):
        audit(bad,m.events,4,(8,8,8,4,4,4))
    bad=copy.deepcopy(tr)
    bad[3]["ages_after"][1]+=1
    with pytest.raises(ValueError):
        audit(bad,m.events,4,(8,8,8,4,4,4))


def test_park_remains_clean_across_low_ticks_and_resumes_no_reinitialization():
    _,tr,_=run("external_w2",(8,8,4,4,8),early_tail=False)
    assert tr[2]["active"]==tr[3]["active"]==2
    assert (tr[2]["q"],tr[3]["q"])==(8,4)
    assert tr[2]["ages_after"][1]==tr[3]["ages_after"][1]
    _,tr,_=run("external_w2",(8,8,4,8),early_tail=False)
    assert tr[3]["ages_before"][1]==tr[2]["ages_before"][1] and tr[3]["ages_after"][1]==2
    assert widths(256,256,[[],[0]])==(2,1)
    with pytest.raises(ValueError):
        widths(256,128,[[],[]])


def test_speculative_done_cannot_commit_against_old_head():
    carry,_,_=run("external_w2",(8,),guard=False,early_tail=True,context_sensitive=True)
    checked,tr,_=run("external_w2",(8,),early_tail=True,context_sensitive=True)
    assert carry[4]==5 and checked[4]==7
    first=next(i for i,e in enumerate(tr) if e["promoted"]==[1])
    assert tr[first]["encoded"]==[0] and tr[first+1]["head"]==1
    assert 0 in tr[first+1]["strict_edits"][0]
