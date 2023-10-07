"""Test coarsen module"""
import numpy as np
import pandas as pd
import pytest
from cem.coarsen import coarsen


@pytest.mark.parametrize(
    "treatment,predictor,expected",
    [
        ([0, 0, 1, 1], [1, 2, 3, 4], [0, 0, 1, 1]),  # all stratum empty
    ],
)
def test_coarsen(treatment, predictor, expected):  # noqa: D103
    df = pd.DataFrame({"treatment": treatment, "predictor": predictor})
    df_coarse = coarsen(df, "treatment", columns=["predictor"])
    print(df_coarse)
    assert np.allclose(pd.factorize(df_coarse["predictor"])[0], expected)
