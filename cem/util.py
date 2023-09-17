"""Utilities for CEM"""


def _missing_continuous(coarsening: dict, is_continuous: dict[str, bool], outcome: str, ignore_outcome: bool = True) -> list[str]:
    """
    Find continuous columns missing from the coarsening schema

    coarsening : dict
        Proposed coarsening schema
    is_continuous : dict[str, bool]
        Whether each column is a continuous variable in the original dataset
    outcome : str
        The outcome variable
    ignore_outcome : bool
        Whether to ignore the outcome variable when checking the coarsening schema for continuous variables
    """
    return [var for var, cont in is_continuous.items() if var not in coarsening and cont and not (var == outcome and ignore_outcome)]
