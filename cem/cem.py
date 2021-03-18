'''Coarsened exact matching for causal inference
(Iacus et al. 2012)
'''

from __future__ import absolute_import

import pandas as pd

from .imbalance import imbalance, get_imbalance_params, _univariate_imbalance

__author__ = "Lewis Bails <lewis.bails@gmail.com>"
__version__ = "0.1.3"


class CEM:
    '''Coarsened Exact Matching

    Parameters
    ----------
    data : DataFrame
    treatment : str
        Name of column containing the treatment variable
    outcome : str
        Name of column containing the outcome variable
    continuous : list
        Names of columns containing continous predictors
    H : int, optional
        The number of bins to use for the continuous variables when calculating imbalance
    measure : str, optional
        Multivariate imbalance measure to use (only L1 and L2 imbalance supported)

    Attributes
    ---------
    data, treatment, outcome, continuous, H: see Parameters
    _bins : array_like
        Array of bin edges
    preimbalance : float
        The multidimensional imbalance of the data prior to matching
    measure : str
        Multivariate imbalance measure

    '''

    def __init__(self,
                 data: pd.DataFrame,
                 treatment: str,
                 outcome: str,
                 continuous: list = None,
                 H: int = None,
                 measure: str = 'l1',
                 lower_H: int = 1,
                 upper_H: int = 10):
        if continuous is None:
            continuous = []
        self.data = data.copy()
        self.treatment = treatment
        self.outcome = outcome
        self.continuous = continuous
        self.measure = measure
        self._bins = None
        self.preimbalance = None

        # if H is not given, find a reasonable value using some heuristic
        self.H = self._find_H(lower_H, upper_H) if H is None else H

        # find bin edges for all column containing continuous data bar the outcome and treatment variable
        df = self.data.drop(columns=self.outcome)
        self._bins = get_imbalance_params(df.drop(columns=self.treatment), self.measure, self.continuous, self.H)

        # calculate the imbalance prior to CEM
        self.preimbalance = imbalance(df, self.treatment, self.measure, self._bins)

    def imbalance(self, coarsening: dict, one_to_many: bool = True) -> float:
        '''Calculate the imbalance remaining after matching the data using some coarsening

        Parameters
        ----------
        coarsening : dict
            Defines the strata.
            Keys are the covariate/column names and values are dict's themselves with keys of "bins" and "cut"
            "bins" is the first parameter to the "cut" method stipulated (i.e. number of bins or bin edges, etc.)
            "cut" is the Pandas method to use for grouping the covariate (only "cut" and "qcut" supported)
        one_to_many : bool
            False limits the matches in a stratum to k:k
            True allows k:n matches in a stratum

        Returns
        -------
        float
            The residual imbalance
        '''

        df = self._data_positive_weights(coarsening, one_to_many)
        return imbalance(df, self.treatment, self.measure, self._bins)

    def univariate_imbalance(self, coarsening: dict, one_to_many: bool = True) -> pd.DataFrame:
        '''Calculate the marginal imbalance remaining after matching the data using some coarsening

        Parameters
        ----------
        coarsening : dict
            Defines the strata.
            Keys are the covariate/column names and values are dict's themselves with keys of "bins" and "cut"
            "bins" is the first parameter to the "cut" method stipulated (i.e. number of bins or bin edges, etc.)
            "cut" is the Pandas method to use for grouping the covariate (only "cut" and "qcut" supported)
        one_to_many : bool
            False limits the matches in a stratum to k:k
            True allows k:n matches in a stratum

        Returns
        -------
        pd.DataFrame
            The residual imbalance for each covariate
        '''
        df = self._data_positive_weights(coarsening, one_to_many)
        return _univariate_imbalance(df, self.treatment, self.measure, self.bins)

    def match(self, coarsening: dict, one_to_many: bool = True) -> pd.Series:
        ''' Perform coarsened exact matching using some coarsening schema and return the weights for each example

        Parameters
        ----------
        coarsening : dict
            Defines the strata.
            Keys are the covariate/column names and values are dict's themselves with keys of "bins" and "cut"
            "bins" is the first parameter to the "cut" method stipulated (i.e. number of bins or bin edges, etc.)
            "cut" is the Pandas method to use for grouping the covariate (only "cut" and "qcut" supported)
        one_to_many : bool
            False limits the matches in a stratum to k:k
            True allows k:n matches in a stratum

        Returns
        -------
        pd.Series
            The weight to use for each example of the provided data given the coarsening provided
        '''
        return match(self.data.drop(columns=self.outcome), self.treatment, coarsening, one_to_many)

    def _find_H(self, lower, upper) -> int:
        print('Calculating H, this may take a few minutes.')
        n_bins = range(lower, upper)
        imb = []
        df = self.data.drop(columns=self.outcome)
        for h in n_bins:
            bins = get_imbalance_params(df.drop(columns=self.treatment), self.measure, self.continuous, h)
            imb_h = imbalance(df, self.treatment, self.measure, bins)
            imb.append(imb_h)
        imb = pd.Series(imb, index=n_bins)
        return (imb.sort_values(ascending=False) <= imb.quantile(.5)).idxmax()

    def _data_positive_weights(self, coarsening, one_to_many) -> pd.DataFrame:
        weights = self.match(coarsening, one_to_many)
        return self.data.drop(columns=self.outcome).loc[weights > 0, :]


def match(data: pd.DataFrame, treatment: str, coarsening: dict, one_to_many=True) -> pd.Series:
    '''Return weights for data given a coursening schema

    Parameters
    ----------
    data : pd.DataFrame
        The data on which we shall perform coarsened exact matching
    treatment : str
        The name of the column in data containing the treatment variable
    coarsening : dict
        Defines the strata.
        Keys are the covariate/column names and values are dict's themselves with keys of "bins" and "cut"
        "bins" is the first parameter to the "cut" method stipulated (i.e. number of bins or bin edges, etc.)
        "cut" is the Pandas method to use for grouping the covariate (only "cut" and "qcut" supported)
    one_to_many : bool
        False limits the matches in a stratum to k:k
        True allows k:n matches in a stratum
    '''
    # coarsen based on supplied coarsening schema
    data_ = coarsen(data.copy(), coarsening)

    # weight data in non-empty strata
    if one_to_many:
        return _weight(data_, treatment)
    else:
        raise NotImplementedError('k:k matching is not yet available. Pull requests are welcome.')
        # TODO: k:k matching using bhattacharya for each stratum, weight is 1 for the control and its treatment pair


def _weight(data, treatment) -> pd.Series:
    '''Weight observations based on global and local (strata) treatment level populations'''
    # only keep stata with examples from each treatment level
    # if the treatment is continuous and was not coarsened, this will almost certainly
    # result in 0 weight for all examples
    gb = list(data.drop(treatment, axis=1).columns.values)
    prematched_weights = pd.Series([0] * len(data), index=data.index)
    matched = data.groupby(gb).filter(lambda x: len(
        x[treatment].unique()) == len(data[treatment].unique()))
    if not len(matched):
        # no strata had all levels of the treatment variable
        return prematched_weights
    global_counts = matched[treatment].value_counts()
    weights = matched.groupby(gb)[treatment].transform(lambda x: _weight_stratum(x, global_counts))
    return weights.add(prematched_weights, fill_value=0)


def _weight_stratum(stratum, M) -> pd.Series:
    '''Calculate weights for observations in an individual stratum'''
    ms = stratum.value_counts()  # local counts for levels of the treatment variable
    T = stratum.max()  # use as "under the policy" level
    return pd.Series([1 if c == T else (M[c] / M[T]) * (ms[T] / ms[c]) for _, c in stratum.iteritems()])


def _cut(col, method, bins) -> pd.Series:
    '''Group values in a column into n bins using some Pandas method'''
    if method == 'qcut':
        return pd.qcut(col, q=bins, labels=False)
    elif method == 'cut':
        return pd.cut(col, bins=bins, labels=False)
    else:
        raise Exception(
            f'"{method}" not supported. Coarsening only possible with "cut" and "qcut".')


def coarsen(data: pd.DataFrame, coarsening: dict) -> pd.DataFrame:
    '''Coarsen data based on schema'''
    df_coarse = data.apply(lambda x: _cut(
        x, coarsening[x.name]['cut'], coarsening[x.name]['bins']) if x.name in coarsening else x, axis=0)
    return df_coarse
