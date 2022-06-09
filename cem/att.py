import statsmodels.api as sm
import pandas as pd


def att_wls(data: pd.DataFrame, weights: pd.Series, treatment: str,
            outcome: str) -> sm.regression.linear_model.RegressionResults:
    '''
    Takes a CEM object and generates an estimated treatment effect using statsmodels WLS

    Parameters
    ----------
    data : pd.DataFrame
        A pandas DataFrame of data (input for CEM)
    weights : pd.Series
        A pandas Series of weights (as returned by the CEM matching function)
    treatment : str
        Name of column in dataframe containing the treatment variable
    outcome : str
        Name of column in dataframe containing the outcome variable

    Returns
    -------
    A statsmodel RegressionResults object
    '''

    X = data[treatment]
    Y = data[outcome]

    X = sm.add_constant(X)

    mod = sm.WLS(Y, X, weights=weights)
    return mod.fit()


def att_weighted_mean(data: pd.DataFrame, weights: pd.Series, treatment: str, outcome: str) -> pd.DataFrame:
    '''

    Parameters
    ----------
    data : pd.DataFrame
        A pandas DataFrame of data (input for CEM)
    weights : pd.Series
        A pandas Series of weights (as returned by the CEM matching function)
    treatment : str
        Name of column in dataframe containing the treatment variable
    outcome : str
        Name of column in dataframe containing the outcome variable

    Returns
    -------
    A pandas DataFrame object

    '''

    tmp = data.copy()
    tmp['weighted_outcome'] = weights * tmp[outcome]
    return tmp.groupby(treatment).agg({'weighted_outcome': 'mean'})
