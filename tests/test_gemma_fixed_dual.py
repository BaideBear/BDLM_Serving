from collections import Counter

import pytest
import torch

from test_gemma_strict_request import B, PROMPT, V, inputs, model
from bdlm_serving.methods.t2t.budget_family import Law
from experiments.t2t.serving_slots.gemma_budget_family import FamilyRequest
from experiments.t2t.serving_slots.gemma_fixed_dual import FixedFamilyRequest, WaveKV
from experiments.t2t.serving_slots.gemma_fixed_dual_eval import audited


@pytest.mark.parametrize('q', [32, 64, 128, 256, 512])
def test_one_canvas_matches_family(model, q):
    kw = dict(seed=19,max_new_tokens=B,law=Law())
    ctrl = FamilyRequest(model,inputs(),**kw)
    req = FixedFamilyRequest(model,inputs(),q=q,**kw)
    with pytest.raises(ValueError, match='fixed offer'):
        req.step(64 if q!=64 else 32)
    while ctrl.status == 'ready':
        c, e = ctrl.step(min(q,B)), req.step()
        if q<=B:
            assert c == e
        else:
            p, = e['parts']
            for k in ('ordinal','phase','rows','dirty_before','prefix','sampled','admitted','edited','revoked'):
                assert p[k] == c[k]
            assert e['positions'] == [len(PROMPT)+i for i in c['rows']]
            assert 0 < e['q'] == c['q'] <= B
        if req.status == 'ready':
            assert torch.equal(ctrl.sampler.ids, req.sampler.ids)
            assert torch.equal(ctrl.sampler.rng.get_state(), req.sampler.rng.get_state())
    assert req.status == 'done' and req.tokens == ctrl.tokens


@torch.inference_mode()
def test_actual_direction_in_both_layers_and_logits(model,monkeypatch):
    extend = WaveKV.extend
    seen = []
    def witnessed(bank,layer,q,k,v,mask):
        ko,vo,m = extend(bank,layer,q,k,v,mask)
        assert bank.positions == bank.live == list(range(len(PROMPT),len(PROMPT)+2*B))
        live = m[..., -2*B:]
        assert not live[...,:B,B:].any() and live[...,B:,:B].all()
        assert live[...,:B,:B].all() and live[...,B:,B:].all()
        seen.append(layer)
        return ko,vo,m
    monkeypatch.setattr(WaveKV,'extend',witnessed)
    def first(change):
        r = FixedFamilyRequest(model, inputs(), q=512, seed=19,max_new_tokens=2*B)
        if change == 'tail':
            spawn = r._spawn
            def altered():
                tail = spawn()
                tail.sampler.ids.add_(1).remainder_(V)
                return tail
            monkeypatch.setattr(r,'_spawn',altered)
        if change == 'head':
            r.sampler.ids.add_(1).remainder_(V)
        logits = []
        with model.register_forward_hook(lambda _,a,out: logits.append(out.logits.detach().clone())):
            e = r.step()
        assert e['q'] == 512 and len(logits) == 1 and not e['encoded']
        assert [p['sampled'] for p in e['parts']] == [list(range(B))]*2
        return logits[0].split(B,dim=1)
    h,t = first(None)
    h1,t1 = first('tail')
    h2,t2 = first('head')
    assert torch.equal(h,h1) and not torch.equal(t,t1)
    assert not torch.equal(h,h2) and not torch.equal(t,t2)
    assert seen == [0,1]*3


def test_real_two_block_ledger_and_promotion(model):
    a = audited(model, inputs(), seed=19, max_new_tokens=2*B)
    assert a['max_decoder_Q']==512 and a['joint_F']>0
    assert len(a['output_ids'])==2*B and a['encoder_Q']==B and a['encoder_F']==1
    assert a['Q']==a['decoder_Q']+B and a['F']==a['decoder_F']+1
    ev=a['trace']; t=next(t for t,e in enumerate(ev) if e['encoded'])
    assert ev[t]['promoted']==1 and ev[t]['parts'][0]['phase']=='verify'
    assert ev[t+1]['parts'][0]['phase']=='rebuild' and ev[t+1]['q']==B
    assert not ev[t+1]['parts'][0]['sampled']
    assert ev[t+1]['parts'][0]['age_before']==ev[t+1]['parts'][0]['age_after']
    assert not ev[-1]['encoded'] and ev[-1]['parts'][0]['phase']=='verify'


def test_eos_discards_paid_tail_without_encoder(model, monkeypatch):
    # Every vocab item is EOS: consume exactly the first verified head argmax.
    monkeypatch.setattr(model.generation_config,'eos_token_id',list(range(V)))
    a=audited(model,inputs(),seed=19,max_new_tokens=2*B)
    assert len(a['output_ids'])==1 and a['encoder_Q']==a['encoder_F']==0
    assert a['decoder_F']==a['F'] and a['decoder_Q']==a['Q']
    assert a['joint_F']>0 and a['trace'][-1]['discarded']==[1]
    assert not any(p['phase']=='verify' and p['ordinal']==1 for e in a['trace'] for p in e['parts'])


def test_warm_promotion_and_new_cold_tail(model):
    # Deliberately resolve block 0 first. Block 1 must survive promotion with
    # its own sampler, but not reuse KV computed before encoding sealed block 0.
    mixed = []
    def head_fast(_, a, kw, out):
        ids = kw['decoder_input_ids']
        pos = kw['decoder_position_ids'][0] - len(PROMPT)
        scores = torch.full_like(out.logits, -64.)
        scores.scatter_(-1, ids.unsqueeze(-1), 64.)
        scores[:, (pos>=B) & (pos<2*B)] = 0.
        out.logits = scores
    with model.register_forward_hook(head_fast,with_kwargs=True):
        r = FixedFamilyRequest(model,inputs(),q=512,seed=19,max_new_tokens=3*B)
        while r.ordinal == 0:
            old_bank = r.bank
            if r.tail is not None:
                tail_rng = r.tail.sampler.rng.get_state().clone()
            e = r.step()
        assert e['encoded'] and e['promoted']==1 and r.bank is not old_bank and not r.bank.live
        assert r.cache.get_seq_length()==len(PROMPT)+B
        assert torch.equal(r.sampler.rng.get_state(),tail_rng)
        assert r.rows.accepted != set(range(B)) and r.sampler.age<48
        r.step()  # Pure new-epoch re-projection. Next choose admits a cold tail.
        # Observe the actual soft embedding going into the SC module, after its
        # masking hook ran, not merely zero raw scores (which softmax to uniform).
        def check(_,a,out):
            mixed.append(a[1].clone())
        with model.model.decoder.self_conditioning.register_forward_hook(check):
            e = r.step()
        # With no dirty head, cold-first can project the tail alone, requiring
        # no SC at all. In that case it is reached through the zero-SC path.
        p = next(p for p in e['parts'] if p['ordinal']==2)
        assert p['sampled']==list(range(B)) and p['age_before']==0
        if mixed:
            assert not mixed[0][0,-B:].count_nonzero()
        while r.status=='ready':
            r.step()
        assert len(r.tokens)==3*B


@pytest.mark.parametrize('num',[0,1])
def test_global_prefix_law_and_three_commits(model,num):
    # Keep a small ambiguous tail in each block, with mostly clean accepted rows.
    # Sampler and entropy/renoise are native; replace only final LM scores.
    def scores(_,a,kw,out):
        ids=kw['decoder_input_ids']; rows=(kw['decoder_position_ids']-len(PROMPT))%B
        result=torch.full_like(out.logits,-64.)
        result.scatter_(-1,ids.unsqueeze(-1),64.)
        result.masked_fill_((rows>=240).unsqueeze(-1),0.)
        out.logits=result
    with model.register_forward_hook(scores,with_kwargs=True):
        a=audited(model,inputs(),seed=19,max_new_tokens=3*B,law=Law(num,8))
    assert len(a['output_ids'])==3*B and a['encoder_F']==2 and a['encoder_Q']==512
    assert a['prefill_Q']==len(PROMPT)
    counts=Counter()
    for e in a['trace']:
        ps=e['parts']; charged=sum(len(p['dirty_before']) for p in ps)
        pref=sum(len(p['prefix']) for p in ps)
        assert pref<=min(512-charged,B)*num//8
        if len(ps)==2 and pref:
            counts['joint']+=1
        counts['total']+=pref
    assert bool(counts['joint'])==bool(num) and bool(counts['total'])==bool(num)
