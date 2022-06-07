import statsmodels.api as sm
import pandas as pd

from .cem import CEM


def att_wls(cem: CEM, treatment: str, outcome: str) -> sm.regression.linear_model.RegressionResults:
    '''
    Takes a CEM object and generates an estimated treatment effect using statsmodels WLS

    Parameters
    ----------
    cem : CEM
        A CEM object that has been through matching and contains weights
    treatment : str
        Name of column in dataframe containing the treatment variable
    outcome : str
        Name of column in dataframe containing the outcome variable

    Returns
    -------
    A statsmodel RegressionResults object
    '''

    X = cem.data[treatment]
    Y = cem.data[outcome]

    X = sm.add_constant(X)

    mod = sm.WLS(Y, X, weights=cem.weights)
    return mod.fit()


def att_weighted_mean(cem: CEM, treatment: str, outcome: str) -> pd.DataFrame:
    '''

    Parameters
    ----------
    cem : CEM
        A CEM object that has been through matching and contains weights
    treatment : str
        Name of column in dataframe containing the treatment variable
    outcome : str
        Name of column in dataframe containing the outcome variable

    Returns
    -------
    A pandas DataFrame object

    '''

    tmp = cem.data.copy()
    tmp['weighted_outcome'] = cem.weights * tmp[outcome]
    return tmp.groupby(treatment).agg({'weighted_outcome': 'mean'})
