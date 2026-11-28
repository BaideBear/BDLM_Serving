import numpy as np

from experiments.t2t.serving_slots.gemma_prefix_seat import PrefixSeat


class Bank:
    def __init__(self, resident):
        self.initialized = np.zeros(1024, dtype=bool)
        self.initialized[list(resident)] = True


def state(rows, *, due=(), ready=4, width=64, start=0, offer=4):
    prior = np.zeros(width, dtype=np.uint8)
    prior[:ready] = 1
    dirty = np.zeros(1024, dtype=bool)
    dirty[list(due)] = True
    return dict(local=np.asarray(rows, dtype=np.int64), prior=prior, width=width,
                dirty=dirty, start=start, offer=offer, ready=ready,
                bank=Bank({1, 3, 4, 5, 7}), tick=2)


def call(rows, **kwargs):
    return PrefixSeat().rows(**state(rows, **kwargs))


def test_exchange_is_budget_neutral_and_preserves_boundary():
    seat = PrefixSeat()
    rows = np.asarray([4, 5, 6, 7], dtype=np.int64)
    out = seat.rows(**state(rows))
    assert list(out) == sorted(set(rows) - {5} | {3})
    assert isinstance(out, np.ndarray)
    assert seat.record["exchanges"][0]["before"] == [4, 5, 6, 7]


def test_full_prefix_or_debt_or_nonfull_offer_are_no_ops():
    for kwargs in (dict(ready=0), dict(due=(5,))):
        out = call(np.asarray([4, 5, 6, 7], dtype=np.int64), **kwargs)
        assert list(out) == [4, 5, 6, 7] and isinstance(out, np.ndarray)
    assert list(call((4, 5, 7))) == [4, 5, 7]


def test_opportunity_survives_noops_but_not_a_success():
    seat = PrefixSeat()
    rows = np.asarray([4, 5, 6, 7], dtype=np.int64)
    assert list(seat.rows(**state(rows, ready=0))) == list(rows)
    assert list(seat.rows(**state(rows))) == sorted(set(rows) - {5} | {3})
    assert list(seat.rows(**state(rows))) == list(rows)
    assert len(seat.record["exchanges"]) == 1


def test_dirty_outside_physical_plan_is_protected_noop():
    out = call(np.asarray([4, 5, 6, 7], dtype=np.int64), due=(8,))
    assert list(out) == [4, 5, 6, 7] and isinstance(out, np.ndarray)
