import pytest

from bdlm_serving.methods.t2t.budget_family import Law, choose


@pytest.mark.parametrize("b", (32, 256))
def test_one_prefix_law_all_five_tiers(b):
    for q in (b//8, b//4, b//2, b, 2*b):
        dirty = {3}
        mask = range(b, 3*b)
        prefix = list(reversed(range(4,b)))
        p = choose(q,b,dirty,mask,prefix)
        k = min(b, q-1)//8
        assert p.dirty == (3,) and p.prefix == tuple(sorted(prefix[:k]))
        assert len(p.rows)==q and len(p.mask)==q-1-k
        assert not p.drain_only
        assert p.rows==tuple(sorted(set(p.dirty+p.mask+p.prefix)))


def test_debt_is_never_lost_and_prefix_cannot_make_an_extra_call():
    p=choose(4,32,{3,4,5,6,7},{8,9},[2,1])
    assert p.rows==(3,4,5,6) and p.drain_only
    assert choose(4,32,{7},{8,9},[2,1]).rows==(7,8,9)
    assert choose(8,32,(),(),[3,2,1]).rows==()
    assert choose(32,32,{7},(),[3,2,1]).drain_only


def test_cold_start_projects_only_actual_debt_and_new_rows():
    p=choose(4,32,{1},{3,4,5},[],cold=(2,3,4,5))
    assert p.cold==(2,3,4) and p.rows==(1,2,3,4) and not p.drain_only
    with pytest.raises(ValueError):
        choose(4,32,{0},{1},[0])
    with pytest.raises(ValueError):
        choose(4,32,(),(),[2,2])


@pytest.mark.parametrize("q", (32,64,128,256))
def test_real_model_fixed_widths_and_commit(model, q):
    import torch
    from contextlib import ExitStack
    from experiments.t2t.serving_slots.gemma_budget_family import FamilyRequest
    from experiments.t2t.serving_slots.gemma_unified_audit import projectors
    from test_gemma_strict_request import inputs
    toks=[]
    with torch.inference_mode(), ExitStack() as stack:
        projs, pos=stack.enter_context(projectors(model))
        stack.enter_context(model.register_forward_hook(
            lambda m,args,result:toks.append(result.logits[0].argmax(-1).tolist())))
        req=FamilyRequest(model,inputs(),seed=19,max_new_tokens=256,law=Law())
        trace=[]
        while req.status=='ready':
            assert len(trace)<500
            trace.append(req.step(q))
    widths=[e['q'] for e in trace]
    assert max(widths)<=q and all(t == widths for t in projs.values())
    assert all(e['offer']==q and len(p)==w for e,p,w in zip(trace,pos,widths,strict=True))
    argmax={i:v for e,ids in zip(trace,toks,strict=True) if e['phase']=='verify'
            for i,v in zip(e['rows'],ids,strict=True)}
    assert req.tokens==[argmax[i] for i in range(256)]


from test_gemma_strict_request import model  # common pinned random-weight Gemma fixture
