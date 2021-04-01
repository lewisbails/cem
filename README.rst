==================================================
cem: Coarsened Exact Matching for Causal Inference
==================================================

.. image:: https://img.shields.io/pypi/v/cem.svg
   :target: https://pypi.org/project/cem/
   :alt: pypi


.. image:: https://img.shields.io/travis/lewisbails/cem.svg
   :target: https://travis-ci.com/lewisbails/cem
   :alt: build


.. image:: https://readthedocs.org/projects/cem-coarsened-exact-matching-for-causal-inference/badge/?version=latest
   :target: https://cem-coarsened-exact-matching-for-causal-inference.readthedocs.io/en/latest/?badge=latest
   :alt: docs

cem is a lightweight library for Coarsened Exact Matching (CEM) and is essentially a poor man's version of the original R-package [1].
CEM is a matching technique used to reduce covariate imbalance which would otherwise lead to treatment effect estimates that are sensitive to model specification.
By removing and/or reweighting certain observations via CEM, one can arrive at treatment effect estimates that are more stable than those found using other matching techniques like propensity score matching.
The L1 and L2 multivariate imbalance measures are implemented as described in [2].
I make no claim to originality and thank the authors for their research.

Get the `code <http://github.com/lewisbails/cem>`_, read the `docs <https://cem-coarsened-exact-matching-for-causal-inference.readthedocs.io/>`_, or `contribute <https://cem-coarsened-exact-matching-for-causal-inference.readthedocs.io/en/latest/contributing.html>`_!

Usage
-----

.. code-block:: python

   from cem import CEM

   boston = load_boston()
   ...
   df

   +----+---------+------+---------+--------+-------+-------+-------+--------+-------+-------+-----------+--------+---------+--------+
   |    |    CRIM |   ZN |   INDUS |   CHAS |   NOX |    RM |   AGE |    DIS |   RAD |   TAX |   PTRATIO |      B |   LSTAT |   MEDV |
   +====+=========+======+=========+========+=======+=======+=======+========+=======+=======+===========+========+=========+========+
   |  0 | 0.00632 |   18 |    2.31 |      0 | 0.538 | 6.575 |  65.2 | 4.09   |     1 |   296 |      15.3 | 396.9  |    4.98 |   24   |
   +----+---------+------+---------+--------+-------+-------+-------+--------+-------+-------+-----------+--------+---------+--------+
   |  1 | 0.02731 |    0 |    7.07 |      0 | 0.469 | 6.421 |  78.9 | 4.9671 |     2 |   242 |      17.8 | 396.9  |    9.14 |   21.6 |
   +----+---------+------+---------+--------+-------+-------+-------+--------+-------+-------+-----------+--------+---------+--------+
   |  2 | 0.02729 |    0 |    7.07 |      0 | 0.469 | 7.185 |  61.1 | 4.9671 |     2 |   242 |      17.8 | 392.83 |    4.03 |   34.7 |
   +----+---------+------+---------+--------+-------+-------+-------+--------+-------+-------+-----------+--------+---------+--------+
   |  3 | 0.03237 |    0 |    2.18 |      0 | 0.458 | 6.998 |  45.8 | 6.0622 |     3 |   222 |      18.7 | 394.63 |    2.94 |   33.4 |
   +----+---------+------+---------+--------+-------+-------+-------+--------+-------+-------+-----------+--------+---------+--------+
   |  4 | 0.06905 |    0 |    2.18 |      0 | 0.458 | 7.147 |  54.2 | 6.0622 |     3 |   222 |      18.7 | 396.9  |    5.33 |   36.2 |
   +----+---------+------+---------+--------+-------+-------+-------+--------+-------+-------+-----------+--------+---------+--------+

   c = CEM(df, "CHAS", "MEDV")

   # "bins" can be an int (number of quantiles or equal width bins) or sequence of scalars (quantiles for "qcut" or bin edges for "cut")
   schema = {'CRIM': {'bins': 3, 'method': 'qcut'},
             'ZN': {'bins': 4, 'method': 'cut'},
             'INDUS': {'bins': 4, 'method': 'cut'},
             'NOX': {'bins': 4, 'method': 'cut'},
             'RM': {'bins': 4, 'method': 'cut'},
             'AGE': {'bins': 5, 'method': 'cut'},
             'DIS': {'bins': 5, 'method': 'cut'},
             'RAD': {'bins': 6, 'method': 'cut'},
             'TAX': {'bins': 4, 'method': 'cut'},
             'PTRATIO': {'bins': 5, 'method': 'cut'},
             'B': {'bins': 3, 'method': 'cut'},
             'LSTAT': {'bins': 2, 'method': 'cut'}}

   # Check the multidimensional (L2) imbalance before and after matching
   c.imbalance() # 0.96
   c.imbalance(schema) # 0.67

   # Get the weights for each example after matching using the coarsening schema
   weights = c.match(schema)
   weights[weights > 0]

   +-----+----------+
   |     |  weights |
   +=====+==========+
   | 142 | 1        |
   +-----+----------+
   | 143 | 0.504762 |
   +-----+----------+
   | 144 | 0.504762 |
   +-----+----------+
   | 147 | 0.504762 |
   +-----+----------+
   | 148 | 0.504762 |
   +-----+----------+
   | 149 | 0.504762 |
   +-----+----------+
   | 150 | 1.2619   |
   +-----+----------+
   | 151 | 1.2619   |
   +-----+----------+
   | 154 | 1        |
   +-----+----------+
   ...


   # ..perform weighted regression or weighted difference of means to find your treatment effect

References
----------

[1] Porro, Giuseppe & King, Gary & Iacus, Stefano. (2009). CEM: Software for Coarsened Exact Matching. Journal of Statistical Software. 30. 10.18637/jss.v030.i09.

[2] Iacus, S. M., King, G., and Porro, G. Multivariate matching methods that are monotonic imbalance bounding. Journal of the American Statistical Association 106, 493 (2011 2011), 345–361.

[3] Iacus, S. M., King, G., and Porro, G. Causal inference without balance checking: Coarsened exact matching. Political Analysis 20, 1 (2012), 1–24.

[4] King, G., and Zeng, L. The dangers of extreme counterfactuals. Political Analysis 14 (2006), 131–159.

[5] Ho, D., Imai, K., King, G., and Stuart, E. Matching as nonparametric preprocessing for reducing model dependence in parametric causal inference. Political Analysis 15 (2007), 199–236.