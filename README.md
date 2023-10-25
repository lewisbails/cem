# cem: Coarsened Exact Matching for Causal Inference

![pypi](https://img.shields.io/pypi/v/cem.svg)
![pytest](https://github.com/lewisbails/cem/actions/workflows/pytest.yml/badge.svg?event=push&branch=master)
![style](https://github.com/lewisbails/cem/actions/workflows/style.yml/badge.svg?event=push&branch=master)


[cem](https://lewisbails.github.io/cem/) is a lightweight library for Coarsened Exact Matching (CEM). CEM is a matching technique used to reduce covariate imbalance, which would otherwise lead to treatment effect estimates that are sensitive to model specification. By removing and/or reweighting certain observations via CEM, one can arrive at treatment effect estimates that are more stable than those found using other matching techniques like propensity score matching. The L1 and L2 multivariate imbalance measures are implemented as described in [2].

## Usage

### Load the data

```python
from cem.match import match
from cem.coarsen import coarsen
from cem.imbalance import L1

import statsmodels.api as sm

boston = load_boston()

O = "MEDV"  # outcome variable
T = "CHAS"  # treatment variable

y = boston[O]
X = boston.drop(columns=O)
```

|    |    CRIM |   ZN |   INDUS |   CHAS |   NOX |    RM |   AGE |    DIS |   RAD |   TAX |   PTRATIO |      B |   LSTAT |   MEDV |
|----|---------|------|---------|--------|-------|-------|-------|--------|-------|-------|-----------|--------|---------|--------|
|  0 | 0.00632 |   18 |    2.31 |      0 | 0.538 | 6.575 |  65.2 | 4.09   |     1 |   296 |      15.3 | 396.9  |    4.98 |   24   |
|  1 | 0.02731 |    0 |    7.07 |      0 | 0.469 | 6.421 |  78.9 | 4.9671 |     2 |   242 |      17.8 | 396.9  |    9.14 |   21.6 |
|  2 | 0.02729 |    0 |    7.07 |      0 | 0.469 | 7.185 |  61.1 | 4.9671 |     2 |   242 |      17.8 | 392.83 |    4.03 |   34.7 |
|  3 | 0.03237 |    0 |    2.18 |      0 | 0.458 | 6.998 |  45.8 | 6.0622 |     3 |   222 |      18.7 | 394.63 |    2.94 |   33.4 |
|  4 | 0.06905 |    0 |    2.18 |      0 | 0.458 | 7.147 |  54.2 | 6.0622 |     3 |   222 |      18.7 | 396.9  |    5.33 |   36.2 |

### Baseline Coarsening

First we coarsen the data in an automatic fashion and calculate a baseline imbalance we wish to improve upon. Be sure to drop the column containing your outcome variable prior to coarsening/matching. `coarsen` optionally takes a list of columns you'd like to auto-coarsen, ignoring the rest.

```python
# coarsen predictor variables
X_coarse = coarsen(X, T, "l1")

# match observations
weights = match(X_coarse, T)

# calculate weighted imbalance, this is our baseline
L1(X_coarse, T, weights)
```

### Informed Coarsening

It's recommended to coarsen using `pandas.cut` and `pandas.qcut`, but you are free to coarsen your predictor variables however you wish.

```python
# coarsen predictor variables
schema = {
   'CRIM': (pd.cut, {'bins': 4}),
   'ZN': (pd.qcut, {'q': 4}),
   'INDUS': (pd.qcut, {'q': 4}),
   'NOX': (pd.cut, {'bins': 5}),
   'RM': (pd.cut, {'bins': 5}),
   'AGE': (pd.cut, {'bins': 5}),
   'DIS': (pd.cut, {'bins': 5}),
   'RAD': (pd.cut, {'bins': 6}),
   'TAX': (pd.cut, {'bins': 5}),
   'PTRATIO': (pd.cut, {'bins': 6}),
   'B': (pd.cut, {'bins': 5}),
   'LSTAT': (pd.cut, {'bins': 5})
}

X_coarse_2 = X.apply(lambda x: schema[x.name][0](x, **schema[x.name][1]) if x.name in schema else x)

# match observations
weights = match(X_coarse_2, T)

# calculate weighted imbalance
L1(X_coarse_2, T, weights)

# we can also calculate the weighted imbalance using the independently coarsened data
L1(X_coarse, T, weights)

# perform weighted regression
model = sm.WLS(y, sm.add_constant(X), weights=weights)
```

## References

[1] Porro, Giuseppe & King, Gary & Iacus, Stefano. (2009). CEM: Software for Coarsened Exact Matching. Journal of Statistical Software. 30. 10.18637/jss.v030.i09.

[2] Iacus, S. M., King, G., and Porro, G. Multivariate matching methods that are monotonic imbalance bounding. Journal of the American Statistical Association 106, 493 (2011 2011), 345–361.

[3] Iacus, S. M., King, G., and Porro, G. Causal inference without balance checking: Coarsened exact matching. Political Analysis 20, 1 (2012), 1–24.

[4] King, G., and Zeng, L. The dangers of extreme counterfactuals. Political Analysis 14 (2006), 131–159.

[5] Ho, D., Imai, K., King, G., and Stuart, E. Matching as nonparametric preprocessing for reducing model dependence in parametric causal inference. Political Analysis 15 (2007), 199–236.
