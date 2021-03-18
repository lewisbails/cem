# cem

[![pypi](https://img.shields.io/pypi/v/cem.svg)](https://pypi.org/project/cem/)
[![build](https://img.shields.io/travis/lewisbails/cem.svg)](https://travis-ci.com/lewisbails/cem)
[![docs](https://readthedocs.org/projects/cem-coarsened-exact-matching-for-causal-inference/badge/?version=latest)](https://cem-coarsened-exact-matching-for-causal-inference.readthedocs.io/en/latest/?badge=latest)



A Python implmentation of Coarsened Exact Matching (CEM).
This is more or less based off the original R-package created by Iacus et al.
I make no claim to originality and thank the authors for their research.

Original paper:
https://gking.harvard.edu/files/abs/cem-plus-abs.shtml


* Free software: MIT license
* Documentation: https://cem-coarsened-exact-matching-for-causal-inference.readthedocs.io.


Usage
--------


```python
from cem import CEM

df

+----+------+------+------+-----+-----+
|    |   X1 |   X2 |   X3 |   T |   Y |
+====+======+======+======+=====+=====+
|  0 |    1 |  0.5 |    1 |   1 |   1 |
+----+------+------+------+-----+-----+
|  1 |    2 |  6.2 |    0 |   0 |   1 |
+----+------+------+------+-----+-----+
|  2 |    3 |  2.4 |    1 |   0 |   0 |
+----+------+------+------+-----+-----+
|  3 |    2 |  6.3 |    0 |   1 |   0 |
+----+------+------+------+-----+-----+
|  4 |    3 |  1.9 |    0 |   1 |   1 |
+----+------+------+------+-----+-----+

c = CEM(df, "T", "Y", ["X1", "X2"], measure='l2')

# "bins" can be an int (number of quantiles or equal width bins) or sequence of scalars (quantiles for "qcut" or bin edges for "cut")
schema = {
        "X1": {"bins": [0, .25, .75, 1], "cut": "qcut"},
        "X2": {"bins": 3, "cut": "cut"},
}

# Check the multidimensional (L2) imbalance before and after matching
c.preimbalance # 0.65
c.imbalance(schema) # 0.36

# Get the weights for each example after matching using the coarsening schema
weights = c.match(schema)

+----+-----------+
|    |   weights |
+====+===========+
|  0 |      1    |
+----+-----------+
|  1 |      0.14 |
+----+-----------+
|  2 |      0.26 |
+----+-----------+
|  3 |      1    |
+----+-----------+
|  4 |      1    |
+----+-----------+

# ..perform weighted regression or weighted difference of means to find your treatment effect
```
Note: Numbers in the example above are just for show.
