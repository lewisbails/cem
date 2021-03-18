
import numpy as np
import pandas as pd
from collections import OrderedDict
from itertools import combinations
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency


def imbalance(data, treatment, measure, bins):
    '''Evaluate multivariate imbalance'''
    if measure in MEASURES:
        return MEASURES[measure](data, treatment, bins)
    else:
        raise NotImplementedError(f'"{measure}" not a valid measure. Choose from {list(MEASURES.keys())}')


def get_imbalance_params(data, measure, continuous=None, H=5) -> list:
    if continuous is None:
        continuous = []
    if measure == 'l1' or measure == 'l2':
        return _bins_for_L(data, continuous, H)
    else:
        raise NotImplementedError('Only params for L variants imbalance available')


def _bins_for_L(data, continuous, H):
    def nbins(n, s): return min(s.nunique(), H) if n in continuous else s.nunique()
    bin_edges = [np.histogram_bin_edges(x, bins=nbins(i, x)) for i, x in data.items()]
    return bin_edges


def _univariate_imbalance(data, treatment, measure, params):
    marginal = {}
    # it is assumed the elements of bins lines up with the data (minus the treatment column)
    for col, bin_ in zip(data.drop(treatment, axis=1).columns, params):
        cem_imbalance = imbalance(data.loc[:, [col, treatment]],
                                  treatment, measure, [bin_])
        d_treatment = data.loc[data[treatment] > 0, col]
        d_control = data.loc[data[treatment] == 0, col]
        if data[col].nunique() > 2:
            stat = d_treatment.mean() - d_control.mean()
            _, p = ttest_ind(d_treatment, d_control, equal_var=False)
            type_ = 'diff'
        else:  # binary variables
            a = data[treatment]
            b = data[col]
            stat, p, _, _ = chi2_contingency(pd.crosstab(a, b))
            type_ = 'Chi2'

        q = [0, 0.25, 0.5, 0.75, 1]
        diffs = d_treatment.quantile(
            q) - d_control.quantile(q) if type_ == 'diff' else pd.Series([None] * len(q), index=q)
        row = {'imbalance': cem_imbalance, 'measure': measure,
               'statistic': stat, 'type': type_, 'P>|z|': p}
        row.update({f'{int(i*100)}%': diffs[i] for i in q})
        marginal[col] = pd.Series(row)
    return pd.DataFrame.from_dict(marginal, orient='index')


def _L1(data, treatment, bins):
    def func(l, r, m, n): return np.sum(np.abs(l / m - r / n)) / 2
    return _L(data, treatment, bins, func)


def _L2(data, treatment, bins):
    def func(l, r, m, n): return np.sqrt(np.sum((l / m - r / n)**2)) / 2
    return _L(data, treatment, bins, func)


def _L(data, treatment, bins, func):
    '''Evaluate Multidimensional Ln score'''
    groups = data.groupby(treatment).groups
    data_ = data.drop(treatment, axis=1).copy()

    try:
        h = {}
        for k, i in groups.items():
            h[k] = np.histogramdd(data_.loc[i, :].to_numpy(), density=False, bins=bins)[0]
        L = {}
        for pair in map(dict, combinations(h.items(), 2)):
            pair = OrderedDict(pair)
            (k_left, k_right), (h_left, h_right) = pair.keys(), pair.values()  # 2 keys 2 histograms
            L[tuple([k_left, k_right])] = func(h_left, h_right, len(groups[k_left]), len(groups[k_right]))

    except Exception as e:
        print(e)
        return 1
    if len(L) == 1:
        return list(L.values())[0]
    return L


MEASURES = {
    'l1': _L1,
    'l2': _L2,
}
