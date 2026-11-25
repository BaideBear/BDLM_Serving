import math

import numpy as np
import pytest

from tools.token_prior import residual_rates


def test_correction_counts_old_selected_rows_only():
    x = np.asarray([[1., 12.], [1., 12.], [1., 33.]])
    steps = [(stage, [], [], [0, 2, 3], set(), [], 1, x, labels) for stage, labels in
             (("selected", {0, 2, 9}), ("verify", {3}), ("bootstrap", {3}))]
    rates = residual_rates([steps], np.array([0.]))
    assert rates == pytest.approx({12: math.log(3 / 2), 33: math.log(1 / 1.5)})
