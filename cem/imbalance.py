from __future__ import annotations
import numpy as np
import pandas as pd
from itertools import combinations
from pandas.api.types import is_numeric_dtype


def _imbalance(data: pd.DataFrame, treatment: str, measure: str, weights: pd.Series = None):
    '''Evaluate multivariate imbalance of a set of observations'''
    if measure.lower() == 'l1':
        return _L1(data, treatment, weights)
    elif measure.lower() == 'l2':
        return _L2(data, treatment, weights)
    else:
        raise NotImplementedError(f'"{measure}" is not a valid multivariate imbalance measure (choose from l1 or l2)')


def _generate_imbalance_schema(data: pd.DataFrame, H=5) -> list[int]:
    schema = {}
    for i, x in data.items():
        if is_numeric_dtype(x):
            schema[i] = ('cut', {"bins": min(x.nunique(), H)})
    return schema


def _L1(data: pd.DataFrame, treatment: str, weights: pd.Series):
    def func(l, r):
        return np.sum(np.abs(l / np.sum(l) - r / np.sum(r))) / 2
    return _L(data, treatment, func, weights)


def _L2(data: pd.DataFrame, treatment: str, weights: pd.Series):
    def func(l, r):
        return np.sum(np.sqrt((l / np.sum(l) - r / np.sum(r))**2)) / 2
    return _L(data, treatment, func, weights)


def _L(data: pd.DataFrame, treatment: str, func, weights: pd.Series = None):
    '''Evaluate multivariate Ln imbalance'''
    df = data.drop(columns=treatment)
    bin_labels = list(df.groupby(list(df.columns)).groups.keys())
    if weights is None:
        weights = pd.Series([1] * len(data), index=data.index)
    bin_counts = {}
    for treatment_level, treatment_group in data.groupby(treatment):
        tg = treatment_group.drop(columns=treatment)
        bin_counts[treatment_level] = tg.groupby(list(tg.columns)).apply(lambda g: weights.loc[g.index].sum()).to_dict()
    L = {}
    for (level_a, a_bin_weight_sums), (level_b, b_bin_weight_sums) in combinations(bin_counts.items(), 2):
        a_counts_ = np.array([a_bin_weight_sums.get(k, 0) for k in bin_labels])
        b_counts_ = np.array([b_bin_weight_sums.get(k, 0) for k in bin_labels])
        L[(level_a, level_b)] = func(a_counts_, b_counts_)
    if len(L) == 1:
        return list(L.values())[0]
    return pd.DataFrame.from_records([k + (v,) for k, v in L.items()], columns=[f'{treatment}_level_a', f'{treatment}_level_b', 'imbalance'])


def _cut(col: str, func: str, func_kwargs: dict) -> pd.Series:
    '''Group values in a column into bins using some Pandas function'''
    if "labels" in func_kwargs or "retbins" in func_kwargs:
        raise ValueError('"labels" and "retbins" arguments are not configurable during coarsening')
    if func == 'qcut':
        return pd.qcut(col, labels=False, retbins=False, **func_kwargs)
    elif func == 'cut':
        return pd.cut(col, labels=False, retbins=False, **func_kwargs)
    else:
        raise ValueError(
            f'"{func}" not supported. Coarsening only possible with "cut" and "qcut".')


def _coarsen(data: pd.DataFrame, coarsening: dict) -> pd.DataFrame:
    '''Coarsen data based on schema'''
    df_coarse = data.apply(lambda x: _cut(x, coarsening[x.name][0], coarsening[x.name][1]) if x.name in coarsening else x, axis=0)
    return df_coarse
