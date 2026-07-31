.. title:: User Guide

.. _user_guide:

==========
User Guide
==========

pyFOCI provides a Python implementation of the **Feature Ordering by Conditional Independence (FOCI)** algorithm.
FOCI is a feature selection method designed to identify a subset of features that are most relevant for modeling a
regression target variable, specifically accounting for nonlinear dependencies.

The algorithm is based on a nonlinear generalization of the partial R² statistic.
This can make pyFOCI particularly useful in scenarios where the relationship between features and the target is strongly nonlinear,
where traditional linear feature selection methods (like Lasso or correlation-based selection) might fail.

How original FOCI Works
-----------------------

FOCI performs **hierarchical forward selection**. The selection process, as described in **Azadkia, M., & Chatterjee, S. (2021)**, works as follows:

First estimators for a conditional dependence coefficient and its unconditional version are defined:

.. math::

    T_n(y,\textbf{Z}|\textbf{X}) :=
    \frac{\sum_{i=1}^n (\min\{R_i, R_{M(i)}\} - \min\{R_i, R_{N(i)}\})}{\sum_{i=1}^n (R_i - \min\{R_i, R_{N(i)}\})}

and

.. math::

    T_n(y,\textbf{Z}) := \frac{\sum_{i=1}^n (n \min\{R_i, R_{M(i)}\} - L_i^2)}{\sum_{i=1}^n L_i(n-L_i)}.

Here :math:`R` is the maximum-rank of the corresponding :math:`y` values: tied values receive the maximal
rank in their tie group. This rank definition is necessary for the paper's proof that :math:`T_n` is a consistent estimator.
:math:`M` and :math:`N` are Euclidean metric nearest neighbors, with ties broken uniformly at random.

The selection process then works as follows:

1. **Initial Step**: The algorithm searches for the single feature that maximizes the (nonlinear) dependence with the target variable :math:`y`, measured by the unconditional :math:`T_n` coefficient.
2. **Iterative Step**: Once a set of features :math:`S_{k-1}` has been selected, FOCI searches for the next feature :math:`j` among the remaining candidates that maximizes the conditional dependence:
   
   :math:`T_n(y, j \mid S_{k-1})`
   
   This means it selects the feature that provides the most "additional" information about the target, given the features already selected.
3. **Stopping Criteria**: The process continues until one of the following conditions is met:

   - The conditional :math:`T_n` of :math:`j` is not positive.
   - All available features have been selected.

How our Implementation Works
----------------------------

We have added a ``max_features`` parameter, and a :math:`T_n` threshold ``min_delta`` to extend the stopping criteria.

Also, the selection process is modified as in the `FOCI R reference implementation <https://cran.r-project.org/package=FOCI>`_
and via the following formula from **Fuchs, S. (2024)**:

.. math::

   T_n(y,\textbf{Z}) =
   1-\frac{3}{n^-1}\sum_{i=1}^n|R_i-R_{N(i)}|+\frac{3}{n^2-1}\left(\sum_{i=1}^n R_{N(i)} + \sum_{i=1}^n R_i - n(n+1)\right).

The Fuchs :math:`T_n` formula is derived for continuous :math:`y`, so using it on data with repeated :math:`y` values
can lead to suboptimal feature selection. Note that the formula contains the offset :math:`-n(n+1)`. It corresponds to the
rank sum equation :math:`\sum_i R_i = n(n+1)/2`, which for non-continuous :math:`y` and maximum rank
does not hold. To alleviate this, we offer a parameter ``rank_method`` to switch from maximum-rank to average-rank:
tied values receive the average rank in their tie group. With average-rank the sum equation always holds.
See :doc:`the rank method comparison example </auto_examples/plot_FOCISelector_average_beats_max>`.

The selection works as follows:

2. **Iterative Step**: Once a set of features :math:`S_{k-1}` has been selected, FOCI searches for the next feature :math:`j` among the remaining candidates that maximizes the **unconditional** dependence:

   :math:`T_n(y, \{j\} \cup S_{k-1})`

   For continuous :math:`y`, this results in the same selection as in original FOCI, because the numerator of the
   conditional :math:`T_n` is the difference of the numerators of two such unconditional :math:`T_n` values.

3. **Stopping Criteria**: The process continues until one of the following conditions is met:

   - The first :math:`T_n` coefficient or the **improvement** in the :math:`T_n` is less than the specified ``min_delta``.
   - All available features have been selected.
   - The number of selected features reaches ``max_features``.

Additionally, we offer a parameter ``nn_tie_breaking`` to switch from the original stochastic :math:`T_n` estimator
to a deterministic version that uses the mean :math:`y` rank of all tied nearest neighbors.

Using ``FOCISelector``
----------------------

The main class provided by the package is :class:`pyFOCI.FOCISelector`. 
It is compatible with the scikit-learn API, meaning it implements ``fit`` and ``transform`` methods and can be used within a :class:`sklearn.pipeline.Pipeline`:

.. code-block:: python

    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from pyFOCI import FOCISelector

    # Create a pipeline that first selects features and then fits a model
    pipeline = Pipeline([
        ('foci', FOCISelector(max_features=5)),
        ('rf', RandomForestRegressor(random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)

For more information on the available parameters and attributes, see the :ref:`api`.

For usage examples, please refer to the :ref:`general_examples`.

References
----------

The pyFOCI implementation is based on the following publications:

* **Azadkia, M., & Chatterjee, S. (2021).** A simple measure of conditional dependence. *The Annals of Statistics*, 49(6), 3070-3102.
* **Fuchs, S. (2024).** Quantifying directed dependence via dimension reduction. *Journal of Multivariate Analysis*, 201, 105266.


Parallel candidate scoring
--------------------------

``FOCISelector`` can score the remaining candidate features in each forward-selection
round in parallel using the scikit-learn convention ``n_jobs``:

.. code-block:: python

    selector = FOCISelector(n_jobs=-1, random_state=0)
    selector.fit(X_train, y_train)

``n_jobs=None`` (the default) and ``n_jobs=1`` score candidates sequentially.
``n_jobs=-1`` uses all processors made available to joblib; other non-zero integer
values request that many worker processes. Parallelism is most useful when many
features remain to be evaluated and computing :math:`T_n` is expensive. Small data
sets may be slower because starting and communicating with worker processes has a
cost.

A fixed integer ``random_state`` assigns deterministic random streams to candidate
features. Consequently, sequential and parallel runs give the same result across
worker counts, including with the default random tie-breaking.

Benchmark locally rather than assuming that ``n_jobs=-1`` is fastest for a particular
data set. From a source checkout, use the Pixi task:

.. code-block:: bash

    OPENBLAS_NUM_THREADS=1 pixi run benchmark-n-jobs

When pyFOCI is installed, run the same benchmark as Python module:

.. code-block:: bash

    OPENBLAS_NUM_THREADS=1 python -m pyFOCI.benchmark

The thread limit prevents each process from creating its own pool of BLAS threads.
If benchmarking in an environment linked against MKL or another OpenMP-based numerical
backend, also set MKL_NUM_THREADS=1 or OMP_NUM_THREADS=1, respectively.

By default, the benchmark compares powers of two below the CPU count and then
``n_jobs=-1`` (all processors). Override the worker counts when needed:

.. code-block:: bash

    OPENBLAS_NUM_THREADS=1 pixi run benchmark-n-jobs --n-jobs 3 5 7
