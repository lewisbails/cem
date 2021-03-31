'''Coarsened exact matching for causal inference'''

from __future__ import absolute_import

import pandas as pd

from .imbalance import imbalance, generate_imbalance_schema, coarsen


class CEM:
    '''Coarsened Exact Matching

    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe containing the observations
    treatment : str
        Name of column in dataframe containing the treatment variable
    outcome : str
        Name of column in dataframe containing the outcome variable
    H : int, optional
        The number of bins to use for the continuous variables when calculating imbalance.
        If None, H will be calculated using a heuristic
        (i.e. The integer value between lower_H and upper_H that produced the median L1 imbalance)
    measure : str, optional
        Multivariate imbalance measure to use (only L1 and L2 imbalance supported)

    Attributes
    ---------
    data, treatment, outcome, continuous, H: see Parameters
    imbalance_schema : dict
        Coarsening schema used to calculate multivariate imbalance
    pre_coarsening_imbalance : float
        The multidimensional imbalance of the data prior to matching
    measure : str
        Multivariate imbalance measure

    '''

    def __init__(self,
                 data: pd.DataFrame,
                 treatment: str,
                 outcome: str,
                 H: int = None,
                 measure: str = 'l1',
                 lower_H: int = 1,
                 upper_H: int = 10):
        self.data = data.copy()
        self.treatment = treatment
        self.outcome = outcome
        self.measure = measure
        if H:
            self.H = H
            self.imbalance_schema = generate_imbalance_schema(self.data.drop(columns=[self.treatment, self.outcome]), self.H)
        else:
            if self.data[treatment].nunique() > 2:
                raise ValueError('Unable to automatically find H if there are more than 2 treatment levels')
            self.H, self.imbalance_schema = self._find_H_and_schema(lower_H, upper_H)

    def imbalance(self, coarsening: dict = None) -> float:
        '''Calculate the multivariate imbalance remaining after matching the data using some coarsening schema

        Parameters
        ----------
        coarsening : dict
            Defines the strata. If None, the returned value is the imbalance prior to performing CEM.
            Keys are the covariate/column names and values are dict's themselves with keys of "bins" and "method"
            "bins" is the first parameter to the binning method stipulated (i.e. number of bins or bin edges, etc.)
            "method" is the Pandas method to use for grouping the covariate (only "cut" and "qcut" supported)
            TODO: This is a very hacky way to pass in a coarsening schema. There must be a better way.

        Returns
        -------
        float
            The residual imbalance
        '''
        weights = (self.match(coarsening) > 0).astype(int) if coarsening else None
        df = coarsen(self.data, self.imbalance_schema)
        return imbalance(df.drop(columns=self.outcome), self.treatment, self.measure, weights)

    def match(self, coarsening: dict = None) -> pd.Series:
        ''' Perform coarsened exact matching using some coarsening schema and return the weights for each observation

        Parameters
        ----------
        coarsening : dict
            Defines the strata.
            Keys are the covariate/column names and values are dict's themselves with keys of "bins" and "method"
            "bins" is the first parameter to the binning method stipulated (i.e. number of bins or bin edges, etc.)
            "method" is the Pandas method to use for grouping the covariate (only "cut" and "qcut" supported)
            TODO: This is a very hacky way to pass in a coarsening schema. There must be a better way.

        Returns
        -------
        pandas.Series
            The weight to use for each observation of the provided data given the coarsening schema provided
        '''
        if coarsening is None:
            coarsening = self.imbalance_schema
        return match(self.data.drop(columns=self.outcome), self.treatment, coarsening)

    def _find_H_and_schema(self, lower: int, upper: int):
        print('Calculating H, this may take a few minutes.')
        n_bins = range(lower, upper)
        imb = {}
        df = self.data.drop(columns=self.outcome)
        for H in n_bins:
            imbalance_schema = generate_imbalance_schema(df.drop(columns=self.treatment), H)
            df_coarse = coarsen(df, imbalance_schema)
            imb_h = imbalance(df_coarse, self.treatment, self.measure)
            imb[H] = [imb_h, imbalance_schema]
        imb = pd.DataFrame.from_dict(imb, orient='index', columns=['imbalance', 'schema'])
        H = (imb['imbalance'].sort_values(ascending=False) <= imb['imbalance'].quantile(.5)).idxmax()
        return H, imb.loc[H, 'schema']


def match(data: pd.DataFrame, treatment: str, coarsening: dict) -> pd.Series:
    '''Return weights for data given a coursening schema

    Parameters
    ----------
    data : pandas.DataFrame
        The data on which we shall perform coarsened exact matching
    treatment : str
        The name of the column in data containing the treatment variable
    coarsening : dict
        Defines the strata.
        Keys are the covariate/column names and values are dict's themselves with keys of "bins" and "method"
        "bins" is the first parameter to the binning method stipulated (i.e. number of bins or bin edges, etc.)
        "method" is the Pandas method to use for grouping the covariate (only "cut" and "qcut" supported)
        TODO: This is a very hacky way to pass in a coarsening schema. There must be a better way.

    Returns
    -------
    pandas.Series
        The weight to use for each observation of the provided data given the coarsening provided
    '''
    # coarsen based on supplied coarsening schema
    df_coarse = coarsen(data, coarsening)

    # weight data in non-empty strata
    return _weight(df_coarse, treatment)


def _weight(data: pd.DataFrame, treatment: str) -> pd.Series:
    '''Weight observations based on global and local (strata) treatment level populations'''
    # only keep stata with examples from each treatment level
    # if the treatment is continuous and was not coarsened, this will almost certainly
    # result in 0 weight for all examples
    gb = list(data.drop(treatment, axis=1).columns)
    prematched_weights = pd.Series([0] * len(data), index=data.index)
    matched = data.groupby(gb).filter(lambda x: len(x[treatment].unique()) == len(data[treatment].unique()))
    if not len(matched):
        # no strata had all levels of the treatment variable
        return prematched_weights
    global_counts = matched[treatment].value_counts()
    weights = pd.concat([_weight_stratum(g[treatment], global_counts) for _, g in matched.groupby(gb)])
    weights = weights.add(prematched_weights, fill_value=0)
    weights.name = 'weights'
    return weights


def _weight_stratum(stratum: pd.Series, M: pd.Series) -> pd.Series:
    '''Calculate weights for observations in an individual stratum'''
    ms = stratum.value_counts()  # local counts for levels of the treatment variable
    T = stratum.max()  # use as "under the policy" level
    stratum_weights = pd.Series([1 if c == T else (M[c] / M[T]) * (ms[T] / ms[c]) for _, c in stratum.iteritems()], index=stratum.index)
    return stratum_weights
