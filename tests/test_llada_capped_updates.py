import pytest
import torch
from types import SimpleNamespace

from bdlm_serving.methods.optimus.fixed import FixedOptimus
from experiments.t2t.serving_slots.llada_capped import audit


class Toy:
    def __call__(self, x, *, position_ids, **kw):
        # Token 2 is admitted; positions divisible by 7 then move to 3 once.
        propose=torch.where(x==0,2,torch.where(position_ids%7==0,3,x))
        logits=torch.zeros((*x.shape,5))
        logits.scatter_(2,propose[...,None],18)
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("mode",["ibs","obs"])
@pytest.mark.parametrize("prompt_len,caps",[(31,(6,)),(41,(8,8,16,16)),(2,(24,))])
def test_capped_long_prompt_dirty_edits_and_cross_block(mode,prompt_len,caps):
    engine=FixedOptimus(Toy(),"llada21",max(caps),mode,physical_caps=caps)
    prompt=torch.ones(1,prompt_len,dtype=torch.long)
    x,events,_=engine.run(prompt,96,17,0,[4])
    assert torch.equal(x[:,:prompt_len],prompt) and not (x[:,prompt_len:]==0).any()
    steps=[e for e in events if e["phase"]!="prefill"]
    assert any(e["edited"] for e in steps) and any(e["dirty_before"] for e in steps)
    if mode=="obs":
        assert any(e["cross_boundary_rows"] for e in steps)
    calls=[dict(rows=e["q_tokens"],physical_rows=e["q_tokens"]) for e in events]
    q=sum(e["q_tokens"] for e in steps)
    audit(dict(events=events,calls=calls,generation_forwards=len(steps),
               generation_query_rows=q,generation_physical_padded_rows=q),caps)


def test_unclipped_fixed32_ibs_matches_original():
    prompt=torch.ones(1,43,dtype=torch.long)
    a=FixedOptimus(Toy(),"llada21",32,"ibs").run(prompt,96,17,0,[4])
    b=FixedOptimus(Toy(),"llada21",32,"ibs",physical_caps=(32,)).run(prompt,96,17,0,[4])
    assert torch.equal(a[0],b[0])
    for x,y in zip(a[1],b[1],strict=True):
        assert all(y[k]==v for k,v in x.items())
