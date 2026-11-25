import numpy as np
import pytest

from tools.budget_index import paired_uncertainty, request_metrics, stratified_draws


def test_within_dataset_pairing_and_pooling():
    keys = [(i, str(i % 3), str(i)) for i in range(30)]
    draws = stratified_draws(keys, draws=100)
    for dataset in range(3):
        assert np.all((draws % 3 == dataset).sum(axis=1) == 10)
    a = np.array([[1, 3 + i % 3, 2, 5 + i % 3] for i in range(30)])
    b = a.copy()
    b[:, 1] -= 1
    result = paired_uncertainty(a, b, draws)
    assert result["only_policy_correct"] == result["only_control_correct"] == 0
    for end in ("lo", "hi"):
        assert result["delta_TPF_" + end] == .5
        assert result["delta_eff_pp_" + end] == pytest.approx(100 / 6)


def test_reject_missing_or_duplicated_request():
    keys = [(i, str(i // 10), str(i)) for i in range(30)]
    rows = [dict(index=i, dataset=ds, task_id=t) for i, ds, t in keys]
    with pytest.raises(ValueError, match="mismatched request"):
        request_metrics(rows[:-1] + rows[:1], keys, {"policy": "example"})
    with pytest.raises(ValueError, match="10 distinct"):
        stratified_draws(keys[:-1])
