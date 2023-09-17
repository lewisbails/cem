def _missing_continuous(coarsening: dict, is_continuous: dict[str, bool], outcome: str, ignore_outcome: bool = True) -> list[str]:
    """Find continuous columns missing from the coarsening schema"""
    return [var for var, cont in is_continuous.items() if var not in coarsening and cont and not (var == outcome and ignore_outcome)]
