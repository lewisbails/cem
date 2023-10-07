"""Coarsening predictor variables for a collection of observations"""
from typing import Sequence, Optional
import pandas as pd
from pandas.api.types import is_numeric_dtype

from cem.imbalance import L1, L2


def coarsen(data: pd.DataFrame, treatment: str, measure: str = "l1", lower: int = 1, upper: int = 10, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Automatic coarsening by binning numeric columns using the number of bins, H, that resulted in the median (unweighted) imbalance over a range of possible values for H.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to coarsen
    treatment : str
        Name of the column containing the treatment level
    measure : str
        Imbalance measure (l1 or l2)
    lower : int
        Minimum value for H
    upper : int
        Maximum value for H
    columns :
        Columns to coarsen
    """
    df = data.copy()

    if columns is None:
        to_coarsen = set(c for c in df.columns if is_numeric_dtype(df[c]))
    else:
        to_coarsen = set(columns)

    if measure == "l1":
        L = L1
    elif measure == "l2":
        L = L2
    else:
        raise ValueError(f"Unknown imbalance measure '{measure}'")

    imb = {}
    for H in range(lower, upper + 1):
        df_coarse = df.apply(lambda x: pd.cut(x, bins=min(x.nunique(), H)) if x.name in to_coarsen else x)
        imb_h = L(df_coarse, treatment)
        if isinstance(imb_h, pd.DataFrame):
            # use the mean imbalance considering all treatment level pairs
            imb_h = imb_h["imbalance"].mean()
        imb[H] = {"imbalance": imb_h, "data": df_coarse}
    imb = pd.DataFrame.from_dict(imb, orient="index")
    H = (imb["imbalance"].sort_values(ascending=False) <= imb["imbalance"].quantile(0.5)).idxmax()
    print(imb.loc[H, "imbalance"])
    return imb.loc[H, "data"]
