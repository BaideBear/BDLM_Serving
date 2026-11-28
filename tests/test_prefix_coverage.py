from collections import Counter

import pytest

from bdlm_serving.methods.t2t.prefix_coverage import spread_prefix


def test_noncontiguous_positions_and_order():
    assert spread_prefix([35, 3, 0, 1, 30], 2) == (3, 35)
    assert spread_prefix([10, 12, 70, 81, 99, 108, 115], 3) == (70, 99, 115)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 8, 31, 32, 63, 128, 256])
def test_partition(n):
    rows = [i * 3 for i in range(n)]
    for k in range(n + 1):
        selected = spread_prefix(rows, k)
        assert len(selected) == len(set(selected)) == k
        assert list(selected) == sorted(selected)
        if k == 0:
            continue
        assert selected[-1] == rows[-1]
        ranks = [v // 3 for v in selected]
        sizes = Counter(b-a for a, b in zip([-1] + ranks, ranks))
        assert sum(size * count for size, count in sizes.items()) == n
        assert max(sizes) - min(sizes) <= 1
        assert set(selected) <= set(rows)


@pytest.mark.parametrize("rows,k", [([], 1), ([1], -1), ([2, 2], 1), ([-2], 0)])
def test_invalid_input(rows, k):
    with pytest.raises(ValueError):
        spread_prefix(rows, k)
