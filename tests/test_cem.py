"""Tests for CEM"""
from contextlib import nullcontext
import pytest
import pandas as pd
import numpy as np
from cem import CEM

size = 50
df = pd.DataFrame(
    {
        "float": np.random.rand(size),
        "int": np.random.randint(0, 10, size),
        "categorical": pd.Series(np.random.randint(0, 4, size), dtype="category"),
        "bool": np.random.randint(0, 2, size, dtype=bool),
        "Y": np.random.randint(0, 2, size, dtype=bool),
    }
)

schema = {
    "float": ('cut', {'bins': 4}),
    "int": ('qcut', {'q': 4}),
}

schema_missing_cont = {
    "int": ('cut', {'bins': 4}),
}


@pytest.mark.parametrize(
    "treatment,H,outcome",
    [
        ("float", None, nullcontext()),
        ("int", None, nullcontext()),
        ("categorical", None, nullcontext()),
        ("bool", None, nullcontext()),
        ("float", 5, nullcontext()),
        ("int", 5, nullcontext()),
        ("categorical", 5, nullcontext()),
        ("bool", 5, nullcontext()),
    ],
)
def test_CEM(treatment, H, outcome):
    """Test CEM object instantiation and initial imbalance check"""
    with outcome:
        c = CEM(df, treatment, "Y", H)
        c.imbalance()
        c.imbalance(schema)
        c.imbalance(schema_missing_cont)
