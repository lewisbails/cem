"""Multidimensional histogram imbalance between two or more collections of observations"""
import pandas as pd
import numpy as np
from typing import Callable, Optional, Union
from itertools import combinations


def L1(data: pd.DataFrame, treatment: str, weights: Optional[pd.Series] = None) -> Union[pd.DataFrame, float]:
    """
    (Weighted) Multidimensional L1 imbalance between groups of observations of differing treatment levels

    Parameters
    ----------
    data : pandas.DataFrame
        Observations
    treatment : str
        Name of column containing the treatment level
    weights : pandas.Series
        Example weights
    """

    def func(tensor_a: np.ndarray, tensor_b: np.ndarray) -> float:
        return np.sum(np.abs(tensor_a / np.sum(tensor_a) - tensor_b / np.sum(tensor_b))) / 2

    return _L(data, treatment, func, weights)


def L2(data: pd.DataFrame, treatment: str, weights: Optional[pd.Series] = None) -> Union[pd.DataFrame, float]:
    """
    (Weighted) Multidimensional L2 imbalance between groups of observations of differing treatment levels

    Parameters
    ----------
    data : pandas.DataFrame
        Observations
    treatment : str
        Name of column containing the treatment level
    weights : pandas.Series
        Example weights
    """

    def func(tensor_a: np.ndarray, tensor_b: np.ndarray) -> float:
        return np.sum(np.sqrt((tensor_a / np.sum(tensor_a) - tensor_b / np.sum(tensor_b)) ** 2)) / 2

    return _L(data, treatment, func, weights)


def _L(data: pd.DataFrame, treatment: str, func: Callable, weights: Optional[pd.Series] = None) -> Union[pd.DataFrame, float]:
    """Evaluate multivariate Ln imbalance, possibly for a treatment variable with more than 2 levels"""
    df = data.drop(columns=treatment)
    strata_cols = list(df.columns)
    strata_labels = list(df.groupby(strata_cols, observed=True).groups)
    if weights is None:
        weights = pd.Series([1] * len(data), index=data.index)
    level_strata_counts = {}
    for level, group in data.groupby(treatment):
        tg = group.drop(columns=treatment)
        level_strata_counts[level] = tg.groupby(strata_cols, observed=True).apply(lambda g: weights.loc[g.index].sum()).to_dict()
    L = {}
    for (level_a, level_a_strata_counts), (level_b, level_b_strata_counts) in combinations(level_strata_counts.items(), 2):
        a_counts_ = np.array([level_a_strata_counts.get(s, 0) for s in strata_labels])
        b_counts_ = np.array([level_b_strata_counts.get(s, 0) for s in strata_labels])
        L[(level_a, level_b)] = func(a_counts_, b_counts_)
    if len(L) == 1:
        return list(L.values())[0]
    return pd.DataFrame.from_records(
        [k + (v,) for k, v in L.items()],
        columns=[f"{treatment}_level_a", f"{treatment}_level_b", "imbalance"],
    )
