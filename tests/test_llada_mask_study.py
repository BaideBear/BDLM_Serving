import pytest

from experiments.t2t.serving_slots.llada_mask_study import REPLAY, study


def test_distinct_registered_studies():
    plans = [study(), *(study(f"fixed{q}") for q in (4, 8, 16, 32))]
    assert len({p.folder for p in plans}) == len(plans)
    assert plans[0].trace == REPLAY and set(plans[0].targets) == {4, 8, 16, 32}
    for q, p in zip((4, 8, 16, 32), plans[1:]):
        assert p.trace == p.targets == (q,)
    with pytest.raises(ValueError):
        study("fixed12")


@pytest.mark.parametrize("q", (4, 8, 16, 32))
def test_fixed_trace_keeps_same_mask_and_physical_work(q, monkeypatch):
    import importlib.util
    from test_llada_offer_masks import PairModel, Tok
    from experiments.t2t.serving_slots import llada_offer_masks as base
    from experiments.t2t.serving_slots import offer_mask_score as scorer
    plan = study(f"fixed{q}")
    monkeypatch.setenv("LLADA_OFFER_MASK_STUDY", f"fixed{q}")
    spec = importlib.util.spec_from_file_location("test_fixed_driver", base.__file__)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner.PLAN == plan and len(runner.ARMS) == 2
    model = PairModel().eval()
    task = dict(index=0, dataset="toy", task_id="t", seed=11)
    a, b = [runner.generate(model, Tok(), task, [5]*32, arm, cap=64)
            for arm in ("causal", f"open{q}")]
    p = runner.pair(a, b, q, trace=plan.trace, cap=64)
    assert not p["exposed"]
    w, _ = scorer.audit_pair(a, b, a, "llada21", scorer.Pair(q, "causal", f"open{q}"),
                             scorer.default_fields("llada21"), plan.trace)
    assert w[0] is None and w[3:] == (0, 0)
