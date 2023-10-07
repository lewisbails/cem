"""Weighted exact matching using coarsened predictor variables"""
import warnings
import pandas as pd


def match(data: pd.DataFrame, treatment: str) -> pd.Series:
    """
    Weight observations based on global and local (strata) treatment level populations

    Only observations from strata with examples from each treatment level will receive a non-zero weight.
    If the treatment column contains continuous values, it is a high likelihood that all examples will receive a weight of zero.

    Parameters
    ----------
    data : pandas.DataFrame
        The data on which we shall perform coarsened exact matching
    treatment : str
        The name of the column in data containing the treatment variable

    Returns
    -------
    pandas.Series
        The weight to use for each observation of the provided data given the coarsening provided
    """
    gb = list(data.drop(columns=treatment).columns)
    prematched_weights = pd.Series([0] * len(data), index=data.index)
    matched = data.groupby(gb).filter(lambda x: x[treatment].nunique() == data[treatment].nunique())

    if not len(matched):
        warnings.warn(
            "No strata had all levels of the treatment variable. All weights will be zero. This usually happens when a continuous variable (including the treatment variable) is not coarsened."
        )
        return prematched_weights
    global_level_counts = matched[treatment].value_counts()
    weights = pd.concat([_weight_stratum(stratum[treatment], global_level_counts) for _, stratum in matched.groupby(gb)])
    weights = weights.add(prematched_weights, fill_value=0)
    weights.name = "weights"
    return weights


def _weight_stratum(local_levels: pd.Series, global_level_counts: pd.Series) -> pd.Series:
    """
    Calculate weights for observations in an individual stratum

    Parameters
    ----------
    local_levels : pandas.Series
        Treatment level for each observation in the given stratum
    global_level_counts : pandas.Series
        Counts for each treatment level over all non-empty strata
    """
    M = global_level_counts
    ms = local_levels.value_counts()  # local counts for levels of the treatment variable
    # choose a level of treatment, T, to denote as "under the policy"
    if local_levels.dtype == "category":
        T = local_levels.cat.as_ordered().max()
    else:
        T = local_levels.max()
    stratum_weights = pd.Series(
        [1 if t == T else (M[t] / M[T]) * (ms[T] / ms[t]) for t in local_levels],
        index=local_levels.index,
    )
    return stratum_weights
