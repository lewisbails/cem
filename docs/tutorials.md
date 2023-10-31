## Boston Housing Data

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

### Automatic Coarsening

First we coarsen the data in an automatic fashion to get a baseline imbalance. Be sure to drop the column containing your outcome variable prior to coarsening/matching. `coarsen` optionally takes a list of columns you'd like to auto-coarsen, ignoring the rest.

```python
# coarsen predictor variables
X_coarse = coarsen(X, T, "l1")

# match observations
weights = match(X_coarse, T)

# calculate weighted imbalance
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

X_coarse = X.apply(lambda x: schema[x.name][0](x, **schema[x.name][1]) if x.name in schema else x)

# match observations
weights = match(X_coarse, T)

# calculate weighted imbalance
L1(X_coarse, T, weights)

# perform weighted regression
model = sm.WLS(y, sm.add_constant(X), weights=weights)
```
