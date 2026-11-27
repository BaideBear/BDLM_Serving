"""Independent CPU contracts; set LLADA21_SOURCE to enable the checkpoint-code test."""
from contextlib import ExitStack
from hashlib import sha256
import importlib
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from bdlm_serving.methods.t2t.llada_budget_request import LLaDABudgetRequest


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Module()
        self.attention.query_key_value = nn.Linear(8,24,bias=False)


class Measured(nn.Module):
    """Real joint attention/KV writes; known token proposals exercise the ledger."""
    def __init__(self, mode='high'):
        super().__init__()
        self.embedding = nn.Embedding(7,8)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Layer(),Layer()])
        self.calls, self.sampler_inputs, self.mode = [], [], mode
        self.eval()

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds, position_ids, attention_mask, past_key_values, **kw):
        pos = position_ids[0]
        mask = attention_mask[0,0]
        keys = torch.arange(mask.shape[-1])
        assert torch.equal(mask,pos[:,None]//32 >= keys//32)
        h = inputs_embeds
        for i, layer in enumerate(self.model.layers):
            q,k,v = layer.attention.query_key_value(h).chunk(3,-1)
            k,v = past_key_values.update(k[:,None],v[:,None],i)
            assert k.shape[-2] == v.shape[-2] == mask.shape[-1]
            h = h + ((q @ k[:,0].mT).masked_fill(~mask,-torch.inf).softmax(-1) @ v[:,0])
        # Recover exactly which actual input token was embedded.
        tok = torch.cdist(inputs_embeds.float(),self.embedding.weight[None].float()).argmin(-1)
        target = torch.where(tok == 0,2,3)
        if self.mode in ('prefix','revocation') and len(self.calls) >= 4:
            target = torch.where(tok == 3,4,torch.where(tok == 4,4,target))
        if self.mode == 'revocation' and len(self.calls) == 2:
            target = torch.where(tok == 2,0,target)
        if self.mode == 'stall':
            target = torch.zeros_like(tok)
        score = torch.full_like(target, 16).float()
        if self.mode in ('prefix','revocation'):
            score = torch.where((pos == 32) | (tok != 0), 16., 1.)
        logits = torch.zeros(1,len(pos),7).scatter(-1,target[...,None],score[...,None])
        self.calls.append((tuple(pos.tolist()),mask.clone(),logits.clone()))
        return SimpleNamespace(logits=logits)

    def _sample_with_temperature_topk_topp(self, logits, **kw):
        assert kw == dict(temperature=0.,top_k=None,top_p=None)
        self.sampler_inputs.append(logits.clone())
        return logits.softmax(-1).max(-1).indices,logits.softmax(-1).max(-1).values


def request(model, prompt=32, cap=69, eos=()):
    return LLaDABudgetRequest(model,torch.ones(1,prompt,dtype=torch.long),
                              seed=42,mask_id=0,eos_ids=eos,max_new_tokens=cap)


def drive(req, model, offers):
    assert len(model.calls)==1 and model.calls[0][0] == tuple(range(req.origin))
    assert req.prefill_record['phase']=='prefill' and not req.records
    debt = set()
    resident = set(range(req.origin))
    consumed = 0
    while req.status=='ready':
        q = offers[len(req.records)%len(offers)]
        old = req.x.clone()
        head = req.head
        e = req.step(q)
        rows, mask, logits = model.calls[-1]
        assert len(model.calls)==len(req.records)+1 and e is req.records[-1] and e['F']==1
        assert tuple(e['rows'])==rows==tuple(sorted(set(rows))) and e['layer_physical_q']==[len(rows)]*2
        assert e['q']==e['physical_q']==len(rows)<=q and tuple(e['dirty_before'])==tuple(sorted(debt))
        assert set(e['dirty'])==set(rows)&debt and len(e['dirty'])==min(q,len(debt))
        assert e['sampled']==sorted(set(e['sampled'])) and set(e['sampled'])<=set(rows)
        if e['sampled']:
            consumed+=1
            assert torch.equal(model.sampler_inputs[-1],logits[:,[rows.index(r) for r in e['sampled']]])
        assert e['sample_clock']==len(model.sampler_inputs)==consumed
        for r in e['prefix']:
            lo = max(req.origin,r//32*32)
            assert r in resident-debt and (old[0,lo:r+1]!=req.mask_id).all()
        assert set(e['cold'])<=set(rows)-resident
        resident.update(rows)
        assert resident==set(range(mask.shape[1]))
        writes = set((old!=req.x)[0].nonzero().flatten().tolist())
        assert writes==set(e['written_absolute'])==set(e['admitted']+e['edited']+e['revoked'])
        assert writes<=set(e['sampled']) and all(r>=max(head,req.origin) for r in writes)
        debt=(debt-set(rows))|writes
        assert debt==set(e['dirty_after'])
        for b in e['retired_blocks']:
            gen=set(range(max(b*32,req.origin),min((b+1)*32,req.end)))
            assert gen<=resident and not gen&debt and (req.x[0,sorted(gen)]!=req.mask_id).all()
    assert req.status=='done' and not debt and not set(req.tokens)&{0}


@pytest.mark.parametrize('q',[4,8,16,32,64,5,37])
def test_fixed_offers_non_aligned_and_short_last_block(q):
    m=Measured()
    req=request(m,prompt=35)
    drive(req,m,[q])
    assert len(req.tokens)==69 and req.x.shape==(1,104)
    assert req.records[0]['rows']==list(range(35,35+min(q,(96 if q>32 else 64)-35)))


def test_downshift_per_layer_debt_eos_and_no_unbilled_tail_discard():
    m=Measured()
    req=request(m,eos=(2,))
    drive(req,m,[64]+[4]*16)
    assert req.records[0]['q']==64 and req.records[0]['dirty_after']==list(range(32,96))
    assert len(req.records)==17 and req.tokens==[2]
    assert req.records[8]['retired_blocks']==[1]
    for i,e in enumerate(req.records[1:]):
        assert e['rows']==list(range(32+i*4,36+i*4)) and e['drain_only'] and not e['sampled']
        assert e['visible_kv']==96


@pytest.mark.parametrize('mode',['prefix','revocation'])
def test_warm_prefix_edit_and_revocation_is_debt(mode):
    m=Measured(mode)
    r=request(m,cap=32)
    drive(r,m,[32]*2+[16]*40)
    if mode=='prefix':
        assert any(set(e['edited']) & set(e['prefix']) for e in r.records)
    else:
        assert any(e['revoked'] and set(e['revoked'])<=set(e['dirty_after']) for e in r.records)


def test_layer_failure_does_not_retire_debt_and_cannot_replay():
    m=Measured()
    r=request(m)
    first=r.step(64)
    debt=set(first['dirty_after'])
    def fail(*args):
        raise RuntimeError('layer failed')
    with m.model.layers[-1].attention.query_key_value.register_forward_pre_hook(fail):
        with pytest.raises(RuntimeError,match='layer failed'):
            r.step(4)
    assert r.status=='failed' and r.dirty==debt and not r.tokens and len(r.records)==1
    with pytest.raises(ValueError):
        r.step(4)


def test_no_silent_mask_completion_or_invalid_offers():
    r=request(Measured('stall'),cap=1)
    for q in (True,3,65,32.):
        with pytest.raises(ValueError):
            r.step(q)
    with pytest.raises(RuntimeError,match='retirement'):
        while r.status=='ready':
            r.step(4)
    assert r.status=='failed' and not r.tokens
    assert len(r.records)==258 and len(r.model.calls)==259  # plus exactly one prompt prefill


@pytest.mark.skipif(not os.environ.get('LLADA21_SOURCE'), reason='Requires local checkpoint Python sources, not weights')
def test_real_checkpoint_decoder_small_random_cpu(monkeypatch):
    from bdlm_serving.runtime.serial import install_llada_rectangular_mask
    root=Path(os.environ['LLADA21_SOURCE'])
    for name,h in [('modeling_llada2_moe','3bbf1e300185b2b28a42f17dbf6bdca57e4129417eb944ec67d699a9826e9cc5'),
                   ('configuration_llada2_moe','618ba95f31bcdd08fd5f92d47d7a3404c5dd5e24bef1c3847f7126025dc263e4')]:
        assert sha256((root/(name+'.py')).read_bytes()).hexdigest()==h
    pkg=ModuleType('_llada21_budget_smoke')
    pkg.__path__=[str(root)]
    monkeypatch.setitem(sys.modules,pkg.__name__,pkg)
    native=importlib.import_module(pkg.__name__+'.modeling_llada2_moe')
    cfg=native.LLaDA2MoeConfig(vocab_size=17,hidden_size=16,head_dim=8,intermediate_size=24,
                  num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,pad_token_id=0,
                  num_experts=2,num_experts_per_tok=1,n_group=1,topk_group=1,num_shared_experts=1,
                  moe_intermediate_size=8,max_position_embeddings=192)
    cfg._attn_implementation='eager'
    with torch.random.fork_rng(devices=[]),torch.device('cpu'):
        torch.default_generator.manual_seed(137)
        model=native.LLaDA2MoeModelLM(cfg).eval()
    install_llada_rectangular_mask(model,allow_selected_cache=True)
    calls=[]
    heads=[]
    with ExitStack() as s:
        s.enter_context(model.register_forward_pre_hook(lambda m,a,kw: calls.append(kw['position_ids'][0].tolist()),
                                                       with_kwargs=True))
        s.enter_context(model.lm_head.register_forward_pre_hook(lambda m,a: heads.append(a[0].shape[1])))
        r=LLaDABudgetRequest(model,torch.ones(1,32,dtype=torch.long),mask_id=16,eos_ids=(),
                             seed=19,max_new_tokens=64)
        while r.status=='ready':
            r.step((64,4,8,16,32)[len(r.records)%5])
    assert len(r.tokens)==64 and 16 not in r.tokens and not r.dirty
    assert calls==[list(range(32))]+[e['rows'] for e in r.records]
    assert heads==[len(rows) for rows in calls]
    assert heads[1]==64 and all(e['q']<=e['offer'] for e in r.records)
    # Change only one side of a genuinely joint projection in a fresh replica.
    @torch.inference_mode()
    def probe(row=None):
        r=LLaDABudgetRequest(model,torch.ones(1,32,dtype=torch.long),mask_id=16,max_new_tokens=64)
        if row is not None:
            r.x[0,row]=5
        return r._project(tuple(range(32,96)),64,'probe')[1]
    base=probe()
    assert torch.equal(base[:,:32], probe(83)[:,:32])
    assert not torch.equal(base[:,32:], probe(43)[:,32:])
