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

How original FOCI Works
-----------------------

FOCI performs **hierarchical forward selection**. The algorithm is based on the **conditional dependence coefficient (CODEC)** introduced in **Azadkia, M., & Chatterjee, S. (2021)**, denoted :math:`T(Y, \textbf{Z} \mid \textbf{X})`. The sample coefficient :math:`T_n` provides a consistent measure bounded between 0 (conditional independence) and 1 (when :math:`Y` is almost surely a measurable function of the variables).

First estimators for the conditional dependence coefficient (conditional CODEC) and its unconditional version are defined:

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

Our implementation provides two selection scoring methods via the ``method`` parameter:

- ``method="fuchs"`` (default): uses the closed-form score from **Fuchs, S. (2024)**:

  .. math::

     T_n(y,\textbf{Z}) =
     1-\frac{3}{n^2-1}\sum_{i=1}^n|R_i-R_{N(i)}|+\frac{3}{n^2-1}\left(\sum_{i=1}^n R_{N(i)} + \sum_{i=1}^n R_i - n(n+1)\right).

- ``method="r_foci"``: uses the :math:`Q_n / S(y)` score corresponding to the `FOCI R reference implementation <https://cran.r-project.org/package=FOCI>`_ based on **Azadkia, M., & Chatterjee, S. (2021)**:

  .. math::

     Q_n(y,\textbf{Z}) = \frac{1}{n^2} \sum_{i=1}^n \left(\min\{R_i, R_{N(i)}\} - \frac{L_i^2}{n}\right)

  and

  .. math::

     S(y) = \frac{1}{n^3} \sum_{i=1}^n L_i(n - L_i),

  where :math:`R_i = \text{rank}(y_i)`, :math:`L_i = \text{rank}(-y_i)`, and :math:`R_{N(i)}` is the target rank of the nearest neighbor of sample :math:`i` in :math:`\textbf{Z}`. The selection score is :math:`Q_n(y, \textbf{Z}) / S(y)` (or 1.0 if :math:`S(y) = 0`).

The selection works as follows:

1. **Initial Step**: The algorithm searches for the single feature :math:`j` that maximizes the selection score on :math:`\{j\}`. If this initial score is less than or equal to the specified ``min_delta``, selection terminates immediately with zero features selected.
2. **Iterative Step**: Once a set of features :math:`S_{k-1}` has been selected, FOCI searches for the next feature :math:`j` among the remaining candidates that maximizes the per-step selection score on :math:`\{j\} \cup S_{k-1}`.
3. **Stopping Criteria**: The process continues until one of the following conditions is met:

   - The **improvement** in the score is less than or equal to the specified ``min_delta``.
   - All available features have been selected.
   - The number of selected features reaches ``max_features``.

``min_delta=0`` corresponds to stopping when the score does not increase (matching the R reference's ``stop=TRUE``), while ``min_delta=None`` disables early stopping (matching ``stop=FALSE``). The per-step selection scores along the path are recorded in the fitted attribute ``score_path_``.

We also offer the parameter ``rank_method`` to configure target rank tie handling: ``rank_method="max"`` (default) is the original definition in **Azadkia & Chatterjee (2021)** and is used in its consistency proof, while ``rank_method="average"`` assigns tied targets their average rank, which also keeps the Fuchs score calibrated (see :ref:`average_vs_max_ranking`).

Additionally, we offer a parameter ``nn_tie_breaking`` to switch from the original stochastic nearest-neighbor selection to alternatives: ``nn_tie_breaking="mean"`` deterministically uses the mean target rank of all tied nearest neighbors, while ``nn_tie_breaking="first"`` lets the nearest-neighbor query return just one nearest neighbor per sample without computing tie sets. These options are available to study their impact on the result, stability and speed of the algorithm (see :doc:`the NN tie-breaking example </auto_examples/plot_FOCISelector_NN_tie_breaking>`).

Difference between Azadkia Paper and R Reference Implementation
---------------------------------------------------------------

Note that ``method="r_foci"`` corresponds to the **CRAN FOCI R reference implementation**, which differs slightly from the theoretical algorithm description in Section 5 of **Azadkia & Chatterjee (2021)**:

- The **original paper algorithm** evaluates candidates in the forward-selection loop using the *conditional* dependence coefficient :math:`T_n(y, j \mid S_{k-1})`, which consists of a *conditional* numerator :math:`Q_n(y, j \mid S_{k-1})` and a *conditional* denominator :math:`S_n(y \mid S_{k-1})`.
- The **R reference implementation** (and pyFOCI's ``method="r_foci"``) instead maximizes the *unconditional* dependence coefficient :math:`T_n(y, \{j\} \cup S_{k-1}) = Q_n(y, \{j\} \cup S_{k-1}) / S(y)` on the growing subset at each step, using the unconditional versions of :math:`Q_n` and :math:`S(y)`.

On continuous data, the candidate feature that maximizes the unconditional coefficient on :math:`\{j\} \cup S_{k-1}` is identical to the one maximizing the conditional coefficient :math:`T_n(y, j \mid S_{k-1})`, because the conditional numerator is the difference of the two unconditional numerators and the denominator :math:`S(y)` depends only on :math:`y`. Using the unconditional form avoids estimating sample-dependent conditional denominators at every step.

.. _average_vs_max_ranking:

Average vs. Max Ranking on Tied Targets
---------------------------------------

On continuous targets without ties, :math:`Q_n / S(y)` and the Fuchs score are identical, and both ranking methods coincide.

When the target variable :math:`y` contains ties or is discrete (e.g. rounded measurements, counts, or binned categories), the choice of ``rank_method`` affects both scoring methods in two distinct ways:

1. **Feature ordering (affects both ``method="fuchs"`` and ``method="r_foci"``):**

   ``rank_method="max"`` assigns all tied observations the group's highest rank, producing a top-heavy weighting that overemphasizes dependencies among high-target samples, while ``rank_method="average"`` assigns the group's midpoint rank and weights all target levels symmetrically. The two orderings can therefore differ, and either method can stop prematurely or select a spurious distractor. Neither is uniformly better, as empirically demonstrated in :doc:`the rank method comparison example </auto_examples/plot_FOCISelector_average_vs_max>`.

2. **Score scaling and baseline calibration (specifically affects ``method="fuchs"``):**

   - The Fuchs formula contains the constant offset :math:`-n(n+1)`, which is mathematically derived under the continuous target rank-sum assumption :math:`\sum_{i=1}^n R_i = n(n+1)/2`.
   - On tied data with ``rank_method="max"``, the sum of maximum ranks is inflated above :math:`n(n+1)/2`. This breaks the offset cancellation, adding an artificial positive shift that frequently pushes Fuchs scores above 1.0 (escaping the unit interval :math:`[0, 1]`).
   - With ``rank_method="average"``, the sum of average ranks **always** equals :math:`n(n+1)/2` exactly, regardless of the number or size of ties. Average ranking thus restores the continuous target rank-sum identity, eliminating the baseline shift and keeping Fuchs scores properly bounded :math:`\le 1.0`.
   - In contrast, ``method="r_foci"`` normalizes by the sample denominator :math:`S(y) = \frac{1}{n^3} \sum L_i(n - L_i)` rather than relying on the continuous target rank-sum assumption, so its scores remain within :math:`[0, 1]` under both ranking methods.

   See :doc:`the score normalization comparison example </auto_examples/plot_FOCISelector_methods>` for an empirical demonstration.

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

All available benchmark options can be displayed via ``--help``:

.. code-block:: text

    usage: benchmark.py [-h] [--samples SAMPLES] [--features FEATURES]
                        [--max-features MAX_FEATURES] [--repeats REPEATS]
                        [--seed SEED] [--method {fuchs,r_foci}]
                        [--n-jobs N_JOBS [N_JOBS ...]]

    Measure local FOCISelector candidate-scoring parallelism. This is
    intentionally a benchmark script, rather than a test: performance varies with
    available CPUs, BLAS threading, and host load.

    options:
      -h, --help            show this help message and exit
      --samples SAMPLES
      --features FEATURES
      --max-features MAX_FEATURES
      --repeats REPEATS
      --seed SEED
      --method {fuchs,r_foci}
                            Selection scoring method: 'fuchs' (default) or 'r_foci'.
      --n-jobs N_JOBS [N_JOBS ...]
                            Worker counts to measure; defaults to powers of two and -1.
