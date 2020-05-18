'''Coarsened exact matching for causal inference'''

from __future__ import absolute_import

import numpy as np
import pandas as pd
from pandas import Series

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
import statsmodels.api as sm

from itertools import combinations, product

from copy import deepcopy
from tqdm import tqdm
from typing import Union

from .imbalance import imbalance, get_imbalance_params, _univariate_imbalance

__author__ = "Lewis Bails <lewis.bails@gmail.com>"
__version__ = "0.1.0"


class CEM:
    '''Coarsened Exact Matching

    Parameters
    ----------
    data : DataFrame
    treatment : str
        The treatment variable in data
    outcome : str
        The outcome variable in data
    continuous : list
        The continuous variables in data
    H : int, optional
        The number of bins to use for the continuous variables when calculating imbalance
    measure : str, optional
        Multivariate imbalance measure to use

    Attributes
    ---------
    data, treatment, outcome, continuous, H: see Parameters
    _bins : array_like
        Array of bin edges
    preimbalance : float
        The imbalance of the data prior to matching
    measure : str
        Multivariate imbalance measure

    '''

    def __init__(self, data: pd.DataFrame, treatment: str, outcome: str, continuous: list = [], H: int = None, measure: str = 'l1'):
        self.data = data.copy()
        self.treatment = treatment
        self.outcome = outcome
        self.continuous = continuous
        self.measure = measure
        self._bins = None
        self.preimbalance = None
        self.H = self._find_H() if H is None else H

        df = self.data.drop(self.outcome, axis=1)
        self._bins = get_imbalance_params(df.drop(self.treatment, axis=1),
                                          self.measure, self.continuous, self.H)
        self.preimbalance = imbalance(df, self.treatment, self.measure, self._bins)

    def _find_H(self):
        print('Calculating H, this may take a few minutes.')
        rows = []
        n_bins = range(1, 10)
        imb = []
        df = self.data.drop(outcome, axis=1)
        for h in n_bins:
            bins = get_imbalance_params(df.drop(self.treatment, axis=1),
                                        self.measure, self.continuous, h)
            imb_h = imbalance(df, self.treatment, self.measure, bins)
            imb.append(imb_h)
        imb = pd.Series(imb, index=n_bins)
        return (imb.sort_values(ascending=False) <= imb.quantile(.5)).idxmax()

    def imbalance(self, coarsening: dict, one_to_many: bool = True) -> float:
        '''Calculate the imbalance remaining after matching the data using some coarsening

        Parameters
        ----------
        coarsening : dict
            Defines the strata.
            Keys are the covariate names and values are dict's themselves with keys of "bins" and "cut"
            "bins" is the first parameter to the "cut" method stipulated (i.e. number of bins or bin edges, etc.)
            "cut" is the Pandas method to use for grouping the covariate (only "cut" and "qcut" supported)
        one_to_many : bool
            Whether to limit the matches in a stratum to k:k, or allow k:n matches.

        Returns
        -------
        float
            The residual imbalance
        '''

        weights = self.match(coarsening, one_to_many)
        df = self.data.drop(self.outcome, axis=1).loc[weights > 0, :]
        return imbalance(df, self.treatment, self.measure, self._bins)

    def univariate_imbalance(self, coarsening: dict = {}, one_to_many: bool = True):
        ''' Calculate the marginal imbalance remaining for each covariate post-matching '''
        weights = self.match(coarsening, one_to_many)
        df = self.data.drop(self.outcome, axis=1).loc[weights > 0, :]
        return _univariate_imalance(df, self.treatment, self.measure, self.bins)

    def match(self, coarsening: dict, one_to_many: bool = True) -> pd.Series:
        ''' Perform CEM using some coarsening schema '''
        return match(self.data.drop(self.outcome, axis=1), self.treatment, coarsening, one_to_many)

    def relax(self, coarsening: dict, relax_vars: 'array_like'):
        ''' Evaluate the residual imbalance and match information for several coarsenings. 
        relax_vars is an iterable of 3-tuples (covariate, bins, method)
        '''
        return Relax(self.data.drop(self.outcome, axis=1), self.treatment, coarsening, relax_vars, self.measure,
                     bins=self.bins)

    def regress(self, coarsening: dict, formula: str, family):
        ''' Match and regress with a single coarsening schema '''
        return match_regress(self.data, self.treatment, self.outcome, coarsening, formula, family)


class Relax:
    '''Summarise the results of progressive coarsening

    Parameters
    ----------
    data: DataFrame
        On which to match
    treatment: str
        Variable defining treatment groups
    coarsening: dict
        Base coarsening schema
    relax_vars: array_like
        3-tuples for defining progressive coarsening
    measure: str
        Multivariable imbalance measure
    continuous: array_like, optional
        Continuous variables in data
    **bins: array_like, optional
        Bin edges for evaluating imbalance

    Attributes
    ----------
    coarsenings: DataFrame
        Imbalance and matching statistics from each coarsening

    '''

    def __init__(self,
                 data: pd.DataFrame,
                 treatment: str,
                 coarsening: dict,
                 relax_vars: 'array_like',
                 measure: str = 'l1',
                 continuous: 'array_like' = [],
                 **kwargs):
        self.total = len(data)
        self.relax_vars = relax_vars
        self.base_coarsening = coarsening
        self.measure = measure
        self.continuous = continuous
        self.__dict__.update(**kwargs)
        self.coarsenings = relax(data, treatment, coarsening, relax_vars, measure, continuous, **kwargs)

    def _plot_multivariate(self, **kwargs):
        '''Plot the percentage of observations matched against the coarsening and annotate with imbalance'''
        s = self.coarsenings.copy()
        t_cols = [col for col in self.coarsenings.columns if 'treatment' in col]
        s['# matched'] = s.loc[:, t_cols].sum(axis=1)
        s['% matched'] = (s['# matched'] / self.total * 100).round(1)
        if len(self.relax_vars) == 1:
            var = self.relax_vars[0][0]
            x = f'{var} bins'
            s[x] = [i[var]['bins'] for i in s['coarsening']]
        else:
            x = 'coarsening'
            s[x] = range(len(s))
            s = s.sort_values('% matched')
        fig, ax = plt.subplots()
        ax = sns.lineplot(x=x, y='imbalance', data=s, style='measure', markers=True, **kwargs)
        fig.set_size_inches(12, 8)
        ax.set_title(f'Multivariate {self.measure} for progressive coarsening')
        for _, row in s.iterrows():
            ax.text(row[x] + 0.15, row['imbalance'] + 0.001, round(row['% matched'], 2), fontsize=10)
        return ax, s

    def _plot_univariate(self, **kwargs):
        s = self.coarsenings.copy()
        x = 'n_bins'
        if len(self.relax_vars) > 1:
            s = s.reset_index('n_bins')
            x = 'coarsening'
            s[x] = range(s['n_bins'].nunique())
            s = s.set_index(x, append=True)

        fig, ax = plt.subplots()
        s = pd.concat({i: j['univariate'].summary for i, j in s.iterrows()})
        s.index.set_names(['var', x, 'covariate'], inplace=True)
        s = s.reset_index()
        ax = sns.lineplot(x=x, y='imbalance', hue='covariate', data=s, ax=ax, **kwargs)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        fig.set_size_inches(12, 8)
        return ax

    def plot(self, name, **kwargs):
        if name in ('multivariate', 'multi'):
            return self._plot_multivariate(**kwargs)
        elif name in ('univariate', 'uni'):
            return self._plot_univariate(**kwargs)
        else:
            raise Exception(f'Unknown plot name "{name}"')


def match(data, treatment, schema, one_to_many=True):
    '''Return weights for data given a coursening schema'''
    # coarsen based on supplied schema
    data_ = coarsen(data.copy(), schema)

    # weight data in non-empty strata
    if one_to_many:
        return _weight(data_, treatment)
    else:
        raise NotImplementedError
        # TODO: k:k matching using bhattacharya for each stratum, weight is 1 for the control and its treatment pair


def _weight(data, treatment):
    '''Weight observations based on global and strata populations'''
    # only keep stata with examples from each treatment level
    gb = list(data.drop(treatment, axis=1).columns.values)
    prematched_weights = pd.Series([0] * len(data), index=data.index)
    matched = data.groupby(gb).filter(lambda x: len(
        x[treatment].unique()) == len(data[treatment].unique()))
    if not len(matched):
        return prematched_weights
    counts = matched[treatment].value_counts()
    weights = matched.groupby(gb)[treatment].transform(lambda x: _weight_stratum(x, counts))
    return weights.add(prematched_weights, fill_value=0)


def _weight_stratum(stratum, M):
    '''Calculate weights for observations in an individual stratum'''
    ms = stratum.value_counts()
    T = stratum.max()  # use as "under the policy" level
    return pd.Series([1 if c == T else (M[c] / M[T]) * (ms[T] / ms[c]) for _, c in stratum.iteritems()])


def _bins_gen(base_coarsening, relax_on):
    '''Individual coarsening schema generator'''
    name = [v[0] for v in relax_on]
    cut_types = [v[-1] for v in relax_on]
    bins = [v[1] for v in relax_on]
    combinations = product(*bins)
    for c in combinations:
        dd = deepcopy(base_coarsening)
        new = {i: {'bins': j, 'cut': k} for i, j, k in zip(name, c, cut_types)}
        dd.update(new)
        yield dd


def relax(data, treatment, coarsening, relax_vars, measure='l1', continuous=[], include_univariate=True, **kwargs):
    '''Match on several coarsenings and evaluate some imbalance measure'''
    assert all([len(x) == 3 for x in relax_vars]
               ), 'Expected variables to relax on as tuple triples (name, iterable, cut method)'
    data_ = data.copy()
    length = np.prod([len(x[1]) for x in relax_vars])

    if 'bins' not in kwargs:
        bins = get_imbalance_params(data_.drop(
            treatment, axis=1), measure, continuous)  # indep. of any coarsening
    else:
        bins = kwargs['bins']

    rows = []
    for coarsening_i in tqdm(_bins_gen(coarsening, relax_vars), total=length):
        weights = match(data_, treatment, coarsening_i)
        nbins = np.prod([x['bins'] if isinstance(x['bins'], int) else len(x['bins']) - 1
                         for x in coarsening_i.values()])
        if len(relax_vars) > 1:
            var = tuple(i[0] for i in relax_vars)
            n_bins = tuple(coarsening_i[i[0]]['bins'] for i in relax_vars)
        elif len(relax_vars) == 1:
            var = relax_vars[0][0]
            n_bins = coarsening_i[var]['bins']
        else:
            var = None
            n_bins = None
        row = {'var': var, 'n_bins': n_bins}
        if (weights > 0).sum():
            d = data_.loc[weights > 0, :]
            if treatment in coarsening_i:
                # continuous treatment binning
                d[treatment] = _cut(d[treatment], coarsening_i[treatment]
                                    ['cut'], coarsening_i[treatment]['bins'])
            score = imbalance(d, treatment, measure, bins)
            vc = d[treatment].value_counts()
            row.update({'imbalance': score,
                        'measure': measure,
                        'weights': weights,
                        'coarsening': coarsening_i,
                        'bins': nbins})
            row.update({f'treatment_{t}': c for t, c in vc.items()})
            if include_univariate:
                row.update({'univariate': _univariate_imbalance(d, treatment, measure, bins)})
        else:
            row.update({'imbalance': 1,
                        'measure': measure,
                        'coarsening': coarsening_i,
                        'weights': None,
                        'bins': nbins})
        rows.append(pd.Series(row))

    return pd.DataFrame.from_records(rows).set_index(['var', 'n_bins'])


def match_regress(data, treatment, outcome, coarsening, formula, family=None):
    '''Regress on 1 or more coarsenings and return the Results for each'''
    data_ = data.copy()
    weights_ = match(data.drop(outcome, axis=1), treatment, coarsening)
    return _regress_matched(data, formula, weights_, family)


def _regress_matched(data, formula, weights, family):
    glm = smf.glm(formula,
                  data=data.loc[weights > 0, :],
                  family=family,
                  var_weights=weights[weights > 0])
    result = glm.fit(method='bfgs', maxiter=1000)
    return result


def _cut(col, method, bins):
    '''Group values in a column into n bins using some Pandas method'''
    if method == 'qcut':
        return pd.qcut(col, q=bins, labels=False)
    elif method == 'cut':
        return pd.cut(col, bins=bins, labels=False)
    else:
        raise Exception(
            f'"{method}" not supported. Coarsening only possible with "cut" and "qcut".')


def coarsen(data, coarsening):
    '''Coarsen data based on schema'''
    df_coarse = data.apply(lambda x: _cut(
        x, coarsening[x.name]['cut'], coarsening[x.name]['bins']) if x.name in coarsening else x, axis=0)
    return df_coarse
