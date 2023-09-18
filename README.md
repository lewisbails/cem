# cem: Coarsened Exact Matching for Causal Inference

![pypi](https://img.shields.io/pypi/v/cem.svg)
![pytest](https://github.com/lewisbails/cem/actions/workflows/pytest.yml/badge.svg?event=push&branch=master)
![style](https://github.com/lewisbails/cem/actions/workflows/style.yml/badge.svg?event=push&branch=master)


[cem](https://lewisbails.github.io/cem/) is a lightweight library for Coarsened Exact Matching (CEM) and is essentially a poor man's version of the original R-package [1]. CEM is a matching technique used to reduce covariate imbalance, which would otherwise lead to treatment effect estimates that are sensitive to model specification. By removing and/or reweighting certain observations via CEM, one can arrive at treatment effect estimates that are more stable than those found using other matching techniques like propensity score matching. The L1 and L2 multivariate imbalance measures are implemented as described in [2]. I make no claim to originality and thank the authors for their research.

## Usage

```python
from cem import CEM

boston = load_boston()
```

|    |    CRIM |   ZN |   INDUS |   CHAS |   NOX |    RM |   AGE |    DIS |   RAD |   TAX |   PTRATIO |      B |   LSTAT |   MEDV |
|----|---------|------|---------|--------|-------|-------|-------|--------|-------|-------|-----------|--------|---------|--------|
|  0 | 0.00632 |   18 |    2.31 |      0 | 0.538 | 6.575 |  65.2 | 4.09   |     1 |   296 |      15.3 | 396.9  |    4.98 |   24   |
|  1 | 0.02731 |    0 |    7.07 |      0 | 0.469 | 6.421 |  78.9 | 4.9671 |     2 |   242 |      17.8 | 396.9  |    9.14 |   21.6 |
|  2 | 0.02729 |    0 |    7.07 |      0 | 0.469 | 7.185 |  61.1 | 4.9671 |     2 |   242 |      17.8 | 392.83 |    4.03 |   34.7 |
|  3 | 0.03237 |    0 |    2.18 |      0 | 0.458 | 6.998 |  45.8 | 6.0622 |     3 |   222 |      18.7 | 394.63 |    2.94 |   33.4 |
|  4 | 0.06905 |    0 |    2.18 |      0 | 0.458 | 7.147 |  54.2 | 6.0622 |     3 |   222 |      18.7 | 396.9  |    5.33 |   36.2 |

```python
c = CEM(df, "CHAS", "MEDV")

# schema are dicts where keys are column names and values are tuples of (panda cut function name, function kwargs)
schema = {
   'CRIM': ('cut', {'bins': 4}),
   'ZN': ('qcut', {'q': 4}),
   'INDUS': ('qcut', {'q': 4}),
   'NOX': ('cut', {'bins': 5}),
   'RM': ('cut', {'bins': 5}),
   'AGE': ('cut', {'bins': 5}),
   'DIS': ('cut', {'bins': 5}),
   'RAD': ('cut', {'bins': 6}),
   'TAX': ('cut', {'bins': 5}),
   'PTRATIO': ('cut', {'bins': 6}),
   'B': ('cut', {'bins': 5}),
   'LSTAT': ('cut', {'bins': 5})
}

# Check the multidimensional (L1) imbalance before and after matching
c.imbalance() # 0.96
c.imbalance(schema) # 0.60

# Get the weights for each example after matching using the coarsening schema
weights = c.match(schema)
weights[weights > 0]
```

|     |   weights |
|-----|-----------|
|   1 |  1.25     |
|   2 |  2.5      |
|  96 |  1.25     |
| 142 |  1        |
| 143 |  0.625    |
| 144 |  0.625    |
| 147 |  0.625    |
| 148 |  0.625    |
| 150 |  2.5      |
| 151 |  2.5      |


## References

[1] Porro, Giuseppe & King, Gary & Iacus, Stefano. (2009). CEM: Software for Coarsened Exact Matching. Journal of Statistical Software. 30. 10.18637/jss.v030.i09.

[2] Iacus, S. M., King, G., and Porro, G. Multivariate matching methods that are monotonic imbalance bounding. Journal of the American Statistical Association 106, 493 (2011 2011), 345–361.

[3] Iacus, S. M., King, G., and Porro, G. Causal inference without balance checking: Coarsened exact matching. Political Analysis 20, 1 (2012), 1–24.

[4] King, G., and Zeng, L. The dangers of extreme counterfactuals. Political Analysis 14 (2006), 131–159.

[5] Ho, D., Imai, K., King, G., and Stuart, E. Matching as nonparametric preprocessing for reducing model dependence in parametric causal inference. Political Analysis 15 (2007), 199–236.
