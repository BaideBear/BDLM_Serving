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


@pytest.mark.parametrize("q",(4,8))
def test_gate_keeps_high_confidence_and_first_fully_visible_call(q):
    def run(arm):
        torch.manual_seed(51)
        m=Measured()
        r=make(m,arm,prompt=35,cap=31)
        drive(r,m,[q])
        return r
    a,b=run("eager"),run("confident")
    assert a.tokens==b.tokens
    assert a.records==[{k:v for k,v in e.items() if k not in ("cold_max_conf","deferred_fallback")}
                       for e in b.records]
    assert any(e.get("cold_max_conf",0)>.9 for e in b.records)
    assert all("deferred_fallback" not in e for e in b.records if e["visible_kv"]>=min(32*(e["block"]+1),66))


def test_gate_veto_is_paid_without_creating_debt_or_reusing_old_proposals():
    class Weak(Measured):
        def _sample_with_temperature_topk_topp(self, logits, **kw):
            self.sampler_inputs.append(logits.clone())
            return torch.full(logits.shape[:-1],2,dtype=torch.long), torch.full(logits.shape[:-1],.1)
    m=Weak()
    r=make(m,"confident",prompt=58,cap=6)
    drive(r,m,[4])
    first,last_cold,followup=r.records[:3]
    assert first["rows"]==first["sampled"]==[58,59,60,61] and first["deferred_fallback"]==[58]
    assert first["written_absolute"]==first["dirty_after"]==[]
    assert last_cold["rows"]==[62,63] and last_cold["admitted"]==[62]
    assert "deferred_fallback" not in last_cold and last_cold["visible_kv"]==64
    assert 58 in followup["sampled"] and followup["dirty_before"]==[62]
