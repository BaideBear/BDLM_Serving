import pytest
import torch

from bdlm_serving.methods.t2t.llada_budget_request import LLaDABudgetRequest
from test_llada_budget_request import Measured, drive


def run(rule,q):
    torch.manual_seed(20)
    m=Measured("prefix")
    r=LLaDABudgetRequest(m,torch.ones((1,58),dtype=torch.long),mask_id=0,
                         max_new_tokens=104,tail_rule=rule)
    drive(r,m,[q])
    return r


@pytest.mark.parametrize("rule",("fill_head","defer_tail"))
def test_cold_tail_rules_are_noop_at_fixed_q32(rule):
    a,b=run("eager",32),run(rule,32)
    assert a.tokens==b.tokens and a.records==b.records


@pytest.mark.parametrize("rule",("fill_head","defer_tail"))
def test_cold_tail_preserves_pre_treatment_and_row_contract(rule):
    a,b=run("eager",64),run(rule,64)
    changed=[e for e in b.records if "tail_intervention" in e]
    assert changed
    e=changed[0]
    tick=e["tick"]
    assert a.records[:tick]==b.records[:tick]
    old=a.records[tick]
    if rule=="defer_tail":
        assert e["rows"]==old["rows"] and e["q"]==old["q"]
        assert not e["sampled"] and not e["written_absolute"]
    else:
        extra=set(e["tail_intervention"]["added"])
        assert extra and not extra.intersection(old["rows"])
        assert e["q"]==old["q"]+len(extra)<=64
        assert all(p//32==e["block"] for p in extra)
        assert set(e["sampled"])==set(old["sampled"])|extra
