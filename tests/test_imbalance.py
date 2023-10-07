"""Test imbalance module"""
import pandas as pd
import numpy as np
import pytest
from cem.imbalance import L1, L2


@pytest.mark.parametrize(
    "func,treatment,predictor,expected",
    [
        (L1, [0, 0, 1, 1], [1, 1, 2, 2], 1),  # level 0: [2, 0], level 1: [0, 2] = 1
        (L1, [0, 0, 1, 1], [1, 2, 1, 2], 0),  # level 0: [1, 1], level 2: [1, 1] = 0
        (L1, [0, 0, 1, 1, 1, 0], [1, 1, 1, 2, 2, 2], 1 / 3),  # level 0: [2, 1], level 1: [2, 1] = 1/3
        (L1, [0, 0, 1, 1], [1, 2, 1, 1], 0.5),  # level 0: [1, 1], level 1: [2, 0] = 0.5
        (L2, [0, 0, 1, 1], [1, 1, 2, 2], 1),  # level 0: [2, 0], level 1: [0, 2] = 1
        (L2, [0, 0, 1, 1], [1, 2, 1, 2], 0),  # level 0: [1, 1], level 2: [1, 1] = 0
        (L2, [0, 0, 1, 1, 1, 0], [1, 1, 1, 2, 2, 2], 1 / 3),  # level 0: [2, 1], level 1: [2, 1] = 1/3
        (L2, [0, 0, 1, 1], [1, 2, 1, 1], 0.5),  # level 0: [1, 1], level 1: [2, 0] = 0.5
    ],
)
def test_L(func, treatment, predictor, expected):  # noqa: D103
    df = pd.DataFrame({"treatment": treatment, "predictor": predictor})
    print(func(df, "treatment"))
    assert np.allclose([func(df, "treatment")], [expected])
