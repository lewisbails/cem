## Boston Housing Data

### Load the data

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

### Create the CEM object

Most cases will require the dataset, the treatment variable name, and the outcome variable name.

```python
c = CEM(df, "CHAS", "MEDV")
```

### Define a schema

Schemas are dicts where keys are column names and values are tuples of (pandas cut function name, function kwargs).

```python
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
```

### Check the imbalance

We can check the multivariate imbalance both before and after coarsening/reweighting to see if the coarsening schema has done what we expected it to do.

```python
c.imbalance() # 0.96
c.imbalance(schema) # 0.60
```

## Calculate the weights

Get the weights for each example after matching using the coarsening schema

```python
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


## Perform regression

We can perform a regression by simply removing zero-weighted examples, or we can use the actual weights themselves.
In either case, given the imbalance was reduced, the effect estimates will be more robust to model specification.
