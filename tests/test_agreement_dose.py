from tools.audit_agreement_dose import selection, tally
import pytest


def step(tick, block, votes, probs, native, ids):
    return dict(tick=tick, block=block, extra=votes, positions=list(range(len(ids))),
                probabilities=probs, native=native, proposals=ids)


def test_matched_dose_does_not_buy_special_or_native_rows():
    s = step(0, 0, [3], [.9, .8, .7, .55, .9], [0, 0, 0, 0, 1], [99, 1, 2, 3, 4])
    assert selection(s, {99})[1:] == ({3}, {1})
    with pytest.raises(ValueError):
        selection(dict(s, extra=[3, 3]), {99})


def test_future_labels_count_only_the_next_native_admission_in_the_same_block():
    steps = [step(0, 0, [0], [.6, .8], [0, 0], [3, 4]),
             step(1, 0, [], [.7, .9], [0, 1], [3, 5]),
             step(2, 1, [], [.9, .8], [1, 1], [3, 4])]
    c = tally(steps, set(), full_tape=True)
    assert (c["dose"], c["replaced_rows"], c["repeat_only_unobserved"],
            c["current_only_different"]) == (1, 1, 1, 1)
    assert not any(k.endswith("unobserved") for k in tally(steps, set(), full_tape=False))


def test_ties_break_by_absolute_position_without_separating_same_sets():
    s = step(0, 0, [0, 1], [.8]*3, [0]*3, [3, 4, 5])
    assert tally([s], set(), full_tape=False)["different_set_steps"] == 0
