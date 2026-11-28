import pytest

from bdlm_serving.methods.t2t.capped_agreement import CappedEvidenceLedger


def observe(ledger, tick, probs, *, pos=(9, 3, 5), ids=(11, 11, 11),
            native=(False, False, False), block=0):
    return ledger.observe(block=block, tick=tick, positions=iter(pos), proposals=iter(ids),
                          probabilities=iter(probs), native=iter(native), excluded_ids={0})


def test_uses_weaker_of_the_two_measurements_and_returns_base_order():
    ledger = CappedEvidenceLedger()
    assert observe(ledger, 0, (.5625, .875, .8125)) == ()
    assert observe(ledger, 1, (.99, .875, .75)) == (3, 5)
    # .125 + .25 fit; the .4375 row is not rescued by its .99 second score.
    assert observe(ledger, 2, (1., .875, .75)) == (9, 3, 5)


def test_equal_cost_tie_absolute_position_and_exact_boundary():
    ledger = CappedEvidenceLedger()
    observe(ledger, 0, (.75, .75, .75))
    assert observe(ledger, 1, (.75, .75, .75)) == (3, 5)
    # An unchosen row still had a real observation on the previous call.
    assert observe(ledger, 2, (.75, .75, .75), native=(False, True, True)) == (9,)


def test_resets_and_native_exclusions_survive_filter():
    ledger = CappedEvidenceLedger()
    observe(ledger, 0, (1., 1., 1.))
    assert observe(ledger, 2, (1., 1., 1.)) == ()
    assert observe(ledger, 3, (1., 1., 1.), block=1) == ()
    assert observe(ledger, 4, (1., 1., 1.), block=1, ids=(11, 0, 12),
                   native=(True, False, False)) == ()


def test_invalid_observation_does_not_advance_either_ledger():
    ledger = CappedEvidenceLedger(allowance=.25)
    observe(ledger, 0, (.625, .875, .625))
    with pytest.raises(ValueError):
        observe(ledger, 1, (.99, .99, .99), native=(False, 2, False))
    assert observe(ledger, 1, (1., 1., 1.)) == (3,)


@pytest.mark.parametrize("allowance", [-1, float("inf"), float("nan")])
def test_invalid_allowance(allowance):
    with pytest.raises(ValueError):
        CappedEvidenceLedger(allowance=allowance)


def test_zero_allowance_can_only_pass_certain_rows():
    ledger = CappedEvidenceLedger(allowance=0)
    observe(ledger, 0, (1., .875, .625))
    assert observe(ledger, 1, (1., 1., .625)) == (9,)
