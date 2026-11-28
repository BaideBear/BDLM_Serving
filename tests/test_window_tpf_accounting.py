import pytest

from experiments.t2t.serving_slots.llada_window_accounting import terms


def test_narrower_with_better_per_q_yield_and_costly_failure_are_not_dropped():
    a=dict(N=8,F=2,Q=16)
    b=dict(N=6,F=2,Q=8)
    r=terms(a,b,8)
    assert (r["width_term"],r["n_per_q_term"],r["delta_tpf"])==(-2.5,1.5,-1.)
    back=terms(b,a,8)
    assert (back["width_term"],back["n_per_q_term"])==(2.5,-1.5)
    assert terms(dict(a,N=0),b,8)["delta_tpf"]==3.
    with pytest.raises(ValueError,match="physical"):
        terms(a,dict(b,Q=17),8)
