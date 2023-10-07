# cem: Coarsened Exact Matching in Python

![pypi](https://img.shields.io/pypi/v/cem.svg)
![pytest](https://github.com/lewisbails/cem/actions/workflows/pytest.yml/badge.svg?event=push&branch=master)
![style](https://github.com/lewisbails/cem/actions/workflows/style.yml/badge.svg?event=push&branch=master)

**Source Code**: [github.com/lewisbails/cem](https://github.com/lewisbails/cem)

**cem** is a lightweight library for Coarsened Exact Matching (CEM), a modern matching method used for causal inference.

Covariate imbalance between treated and control groups can lead to treatment effect estimates that are sensitive to model specification. CEM aims to reduce this imbalance by temporarily discretising covariates using a multidimensional grid, removing strata that do not contain all levels of the treatment variable, and reweighting the remaining examples.

## Getting started

```bash
pip install cem
```

## Acknowledgements

This package borrows much from the original R-package, [cem](https://cran.r-project.org/web/packages/cem/index.html), I make no claim to originality and thank Stefano M. Iacus, Gary King, Giuseppe Porro, and Richard Nielsen for their research.

[1] Porro, Giuseppe & King, Gary & Iacus, Stefano. (2009). CEM: Software for Coarsened Exact Matching. Journal of Statistical Software. 30. 10.18637/jss.v030.i09.

[2] Iacus, S. M., King, G., and Porro, G. Multivariate matching methods that are monotonic imbalance bounding. Journal of the American Statistical Association 106, 493 (2011 2011), 345–361.

[3] Iacus, S. M., King, G., and Porro, G. Causal inference without balance checking: Coarsened exact matching. Political Analysis 20, 1 (2012), 1–24.

[4] King, G., and Zeng, L. The dangers of extreme counterfactuals. Political Analysis 14 (2006), 131–159.

[5] Ho, D., Imai, K., King, G., and Stuart, E. Matching as nonparametric preprocessing for reducing model dependence in parametric causal inference. Political Analysis 15 (2007), 199–236.
