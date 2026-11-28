from tools.ibs_frontier_exposure import swap_if_safe


def try_swap(**changes):
    args = dict(pos=[1, 2, 3, 4], masks=[5, 6], dirty={4},
                initialized=set(range(5)), resolved=set(range(5)),
                prompt=2, origin=0, q=4)
    args.update(changes)
    return swap_if_safe(**args)


def test_only_replace_rightmost_clean_donor_and_keep_actual_q():
    assert try_swap() == (3, 5)
    assert try_swap(dirty={3, 4}) == (2, 5)
    assert try_swap(dirty={2, 3, 4}) is None  # remaining row is prompt
    assert try_swap(dirty={1, 2, 3, 4}) is None


def test_no_donor_on_first_visit_or_hole_in_extended_attention():
    assert try_swap(initialized={0, 1, 4}) is None
    assert try_swap(masks=[6]) is None  # no KV has ever been built at 5
    assert try_swap(masks=[6], initialized=set(range(6)), resolved=set(range(6))) == (3, 6)


def test_no_change_when_baseline_already_has_progress_or_spare_capacity():
    assert try_swap(pos=[1, 2, 3, 5]) is None
    assert try_swap(q=8) is None
    assert try_swap(masks=[]) is None
