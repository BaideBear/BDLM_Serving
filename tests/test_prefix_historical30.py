import pytest
from experiments.t2t.prefix_window.historical30 import pack30, identity


def inputs():
    rows = [dict(index=i, dataset="test", task_id=str(i), seed=i+10, N=8, F=2,
                 Q=4, physical_Q=6, correct=1) for i in range(30)]
    return rows, {identity(r): r["seed"] for r in rows}


def test_work_keeps_both_historical_denominators():
    rows, ids = inputs()
    m, logical = pack30(rows, ids, historical=True)
    assert sum(v[2] for v in m.values()) == 180 and logical == 120


def test_reject_swapped_seed_duplicate_and_missing():
    rows, ids = inputs()
    with pytest.raises(ValueError, match="not the frozen"):
        pack30(rows[:-1], ids)
    with pytest.raises(ValueError, match="duplicate"):
        pack30(rows[:-1] + rows[:1], ids)
    rows[0]["seed"] += 1
    with pytest.raises(ValueError, match="Seed mismatch"):
        pack30(rows, ids)
