"""
This is a module to be used as a reference for building other modules
"""

# Authors: scikit-learn-contrib developers
# License: BSD 3 clause

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import validate_data  # , check_is_fitted


# Note that the mixin class should always be on the left of `BaseEstimator` to ensure
# the MRO works as expected.
class TemplateTransformer(TransformerMixin, BaseEstimator):
    """An example transformer that returns the element-wise square root.

    For more information regarding how to build your own transformer, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "demo_param": [str],
    }

    def __init__(self, demo_param="demo"):
        self.demo_param = demo_param

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        X = validate_data(self, X, accept_sparse=True)

        # Return the transformer
        return self

    def transform(self, X):
        """A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Since this is a stateless transformer, we should not call `check_is_fitted`.
        # Common test will check for this particularly.

        # Input validation
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        X = validate_data(self, X, reset=False, accept_sparse=True)
        return np.sqrt(np.clip(X, 0, None))

    def __sklearn_tags__(self):
        # This is a quick example to show the tags API:\
        # https://scikit-learn.org/dev/developers/develop.html#estimator-tags
        # Here, our transformer does not do any operation in `fit` and only validate
        # the parameters. Thus, it is stateless.
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        tags.input_tags.sparse = True
        return tags
