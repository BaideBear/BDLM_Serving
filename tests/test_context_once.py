from types import SimpleNamespace

import pytest

from bdlm_serving.methods.t2t.context_gate import CUTOFF, ONCE, crowded_plan, select
from bdlm_serving.methods.t2t.protocol import Arm
from bdlm_serving.methods.t2t.seat_audit import SeatAudit


def screen(masks, fresh=(), score=0):
    return select(range(32), [i not in masks for i in range(32)], fresh, [-1]*32, (), 17,
                  8, 'hard_context_gate', 13, 2,
                  lambda old, _: [score if i==30 else -99 for i in old])[2]


@pytest.mark.parametrize('name', ONCE)
def test_one_crowded_transition(name):
    a = SeatAudit(name)
    assert Arm.parse(f'8:0:7:{name}',32).edit == 1
    for spec,width in [('8:0:8',32),('4:0:3',32),('8:0:7',256)]:
        with pytest.raises(ValueError):
            Arm.parse(f'{spec}:{name}',width)
    for ev in (screen({2,4}), screen(set(range(16)), score=-99),
               screen(set(range(16,32)), range(10))):
        assert crowded_plan(ev,13,2,17) is None
        assert a.apply_plan(ev['base'], None) == (ev['base'], [])
        assert not a.used
    m = screen(set(range(16)), score=CUTOFF)
    plan = crowded_plan(m,13,2,17)
    picked, extra = a.apply_plan(m['base'],plan)
    assert a.used and plan['victim']==7 and plan['candidate']==30 and 0 in picked
    assert picked == plan[{'base':'base','rank':'targeted','random':'random'}[name.split('_')[-1]]]
    assert extra == sorted(set(picked)-set(plan['base']))
    with pytest.raises(ValueError):
        a.apply_plan(m['base'],plan)


def test_after_anchor_no_ranking_and_no_reserve():
    from bdlm_serving.methods.t2t.llada_sparse import LLaDASparseKernel
    a = SeatAudit(ONCE[1])
    a.used = True
    h = SimpleNamespace(fresh={9})
    s = SimpleNamespace(history=h, seat_audit=a, events=[], last_write=[-1]*32,
                        last_query=[-1]*32, selection_seed=13, previous_strict=set())
    kernel = SimpleNamespace(policy=ONCE[1], q_budget=8, fresh_slots=0, mask_slots=7)
    result = LLaDASparseKernel._hard_pick(kernel,s,range(32),[i==9 or i>=12 for i in range(32)],[0]*32)
    assert result == ([0,1,2,3,4,5,6,9], [])
