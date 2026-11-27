import pytest
import torch

from bdlm_serving.methods.t2t.llada_low_start import LowStartRequest
from test_llada_budget_request import Measured, drive


def make(model, arm, prompt=35, cap=37, eos=()):
    return LowStartRequest(model,torch.ones((1,prompt),dtype=torch.long),mask_id=0,
                           max_new_tokens=cap,eos_ids=eos,arm=arm)


@pytest.mark.parametrize("q", (4,8))
def test_build_is_paid_and_logits_are_fresh(q):
    m=Measured()
    r=make(m,"resident",eos=(2,))
    drive(r,m,[q])
    build=[e for e in r.records if e["phase"]=="build"]
    assert [p for e in build for p in e["rows"]] == list(range(35,64))
    assert all(not e["sampled"] and not e["dirty_after"] for e in build)
    first=next(e for e in r.records if e["sampled"])
    assert first["rows"]==list(range(35,35+q)) and first["visible_kv"]==64
    assert r.tokens==[2] and r.records[-1]["retired_blocks"]==[1]


def test_single_pass_and_short_last_block_match():
    torch.manual_seed(61)
    m=Measured()
    a=make(m,"eager",cap=34)
    drive(a,m,[32])
    torch.manual_seed(61)
    m=Measured()
    b=make(m,"resident",cap=34)
    drive(b,m,[32])
    assert a.tokens==b.tokens and a.records==b.records


@pytest.mark.parametrize("q",(4,8))
def test_credit_is_real_warm_projection(q):
    class SlowMask(Measured):
        def _sample_with_temperature_topk_topp(self, logits, **kw):
            self.sampler_inputs.append(logits.clone())
            return torch.full(logits.shape[:-1],2,dtype=torch.long), torch.full(logits.shape[:-1],.1)
    m=SlowMask()
    r=make(m,"credit",prompt=32,cap=64)
    drive(r,m,[q])
    assert any(e["prefix"] for e in r.records)
    assert all(len(e["prefix"])<=1 and (not e["prefix"] or e["mask"]) for e in r.records)
