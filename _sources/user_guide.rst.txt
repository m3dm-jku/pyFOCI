.. title:: User Guide : contents

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

Also, the selection process, modified as in the `FOCI R reference implementation <https://cran.r-project.org/package=FOCI>`_
and via **Fuchs, S. (2024)** , works as follows:

2. **Iterative Step**: Once a set of features :math:`S_{k-1}` has been selected, FOCI searches for the next feature :math:`j` among the remaining candidates that maximizes the **unconditional** dependence:
   
   :math:`T_n(y, j ∪ S_{k-1})`
   
   This results in the same selection as in original FOCI, because the numerator of the conditional :math:`T_n` is the difference
   of the numerators of two such unconditional :math:`T_n` values.
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

