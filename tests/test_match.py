"""Test match module"""
import pytest
import pandas as pd
import numpy as np
from cem.match import match


@pytest.mark.parametrize(
    "treatment,predictor,expected",
    [
        ([0, 0, 1, 1], [1, 2, 3, 4], [0, 0, 0, 0]),  # all stratum empty
        ([0, 0, 1, 1], [1, 2, 1, 2], [1, 1, 1, 1]),  # all stratum with both levels
        ([0, 0, 1, 1, 1, 0], [1, 1, 1, 2, 2, 2], [0.5, 0.5, 1, 1, 1, 2]),  # all stratum with both levels, different sizes
        ([0, 0, 1, 1], [1, 2, 1, 1], [1, 0, 1, 1]),  # 1 stratum with both levels
    ],
)
def test_match(treatment, predictor, expected):  # noqa: D103
    df = pd.DataFrame({"treatment": treatment, "predictor": predictor})
    weights = match(df, "treatment")
    print(weights)
    assert np.allclose(weights, expected)
