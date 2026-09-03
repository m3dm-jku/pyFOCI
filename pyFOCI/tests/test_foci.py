"""Pytest for FOCI variable selection using a deterministic synthetic dataset."""

# Authors: Robert Pollak <robert.pollak@jku.at>
# License: BSD 3 clause

import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils._testing import assert_allclose

from pyFOCI import FOCISelector
from pyFOCI._foci import (
    _nn_first_based,
    _nn_grouping_based,
    _nn_radius_based,
    _Qn,
    _rank,
    _S_y,
    _score_candidate,
    _Tn,
)


def make_data(n: int = 100, p: int = 30, seed: int = 0):
    """
    Create a deterministic small dataset for feature selection tests,
    with n entries and p features per entry.
    """
    random_state = np.random.RandomState(seed)
    X = random_state.normal(size=(n, p))
    X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    y = (
        X_df.iloc[:, 0] * X_df.iloc[:, 1]
        + np.sin(X_df.iloc[:, 0] * X_df.iloc[:, 2])
        + X_df.iloc[:, 3] ** 2
    )
    return X_df, y.to_numpy()


def collect_nn_ties(nn_func, X):
    """
    Collect nearest-neighbor tie sets produced by an NN traversal helper.

    The NN helper API aggregates each tie set immediately and returns the
    aggregated values. These tests still need to inspect the tie sets directly,
    so this helper uses a test-only aggregator that records each tie set and
    returns a dummy scalar aggregation value.
    """
    nn_ties_by_sample = []

    def record_ties(nn_ties):
        nn_ties_by_sample.append(np.asarray(nn_ties, dtype=int).copy())
        return 0.0

    aggregated = nn_func(X, record_ties)

    assert isinstance(aggregated, np.ndarray)
    assert aggregated.shape == (np.asarray(X).shape[0],)
    assert aggregated.dtype.kind == "f"
    assert len(nn_ties_by_sample) == np.asarray(X).shape[0]

    return nn_ties_by_sample


def test_default_stopping_and_transform():
    """
    Default early stopping (min_delta=0) at small n selects only a few variables,
    includes column index 3, and transform returns the selected columns in the
    same order.
    """
    X_df, y = make_data(n=300, p=40, seed=0)

    selector = FOCISelector(random_state=0)
    selector.fit(X_df, y)

    names = selector.get_feature_names_out()

    # Expect only a few variables selected
    assert len(names) <= 4

    # Strong marginal effect should be present
    assert "x3" in names

    # Check transform output conforms to scikit-learn conventions
    X_trans = selector.transform(X_df)
    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == X_df.shape[0]
    assert X_trans.shape[1] == len(names)

    # Build expected output using reported names
    expected = X_df.loc[:, names].to_numpy()

    assert_allclose(X_trans, expected)


def test_min_delta_zero_may_select_none_on_independent_data():
    """
    With min_delta=0 (default), on data where y is independent of X,
    the selector can select no features if the best initial Tn <= 0.
    """
    # Small dataset with y independent of X
    random_state = np.random.RandomState(0)
    X = random_state.normal(size=(10, 1))
    y = random_state.normal(size=10)

    selector = FOCISelector(random_state=0, min_delta=0).fit(X, y)

    # With early stopping enabled, zero features may be selected
    assert selector.support_mask_.sum() == 0
    assert len(selector.score_path_) == 0


def test_min_delta_none_ignores_early_stopping_and_selects_up_to_max():
    """
    With min_delta=None, early stopping is ignored and features are selected
    up to max_features even on independent data.
    """
    random_state = np.random.RandomState(0)
    X = random_state.normal(size=(20, 5))
    y = random_state.normal(size=20)

    selector = FOCISelector(random_state=0, min_delta=None, max_features=3).fit(X, y)

    assert selector.support_mask_.sum() == 3
    assert len(selector.score_path_) == 3


def test_min_delta_enforces_gap():
    """
    With a positive min_delta, consecutive cumulative Tn values must improve
    by more than min_delta; the first selected Tn must exceed min_delta.
    """
    X_df, y = make_data(n=200, p=10, seed=0)
    min_delta = 0.03

    selector = FOCISelector(random_state=0, min_delta=min_delta).fit(X_df, y)
    tn = selector.score_path_
    assert tn.size > 1  # precondition for testing a delta

    assert tn[0] > min_delta

    diffs = np.diff(tn)
    assert np.all(diffs > min_delta)


def test_standardize_none():
    """
    standardize=None is also accepted.
    """
    X_df, y = make_data(n=100, p=10, seed=0)

    FOCISelector(random_state=0, standardize=None).fit(X_df, y)


def test_rank_method_max_is_default():
    """
    rank_method="max" is the default and should match an explicit max-rank fit.
    """
    X_df, y = make_data(n=100, p=10, seed=0)

    selector_default = FOCISelector(random_state=0, min_delta=None, max_features=3).fit(
        X_df, y
    )
    selector_max = FOCISelector(
        random_state=0, min_delta=None, max_features=3, rank_method="max"
    ).fit(X_df, y)

    np.testing.assert_array_equal(
        selector_default.selected_indices_,
        selector_max.selected_indices_,
    )
    assert_allclose(
        selector_default.score_path_,
        selector_max.score_path_,
    )


def test_rank_method_average_is_accepted():
    """
    rank_method="average" is accepted and produces a valid fitted selector.
    """
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 0.0],
            [4.0, 1.0],
        ]
    )
    y = np.array([1.0, 1.0, 2.0, 3.0, 3.0])

    selector = FOCISelector(
        random_state=0,
        min_delta=None,
        max_features=1,
        rank_method="average",
    ).fit(X, y)

    assert selector.support_mask_.shape == (X.shape[1],)
    assert selector.support_mask_.sum() == 1
    assert selector.score_path_.shape == (1,)
    assert np.isfinite(selector.score_path_[0])


def test_rank_method_invalid_raises():
    random_state = np.random.RandomState(0)
    X = random_state.normal(size=(20, 3))
    y = random_state.normal(size=20)

    sel = FOCISelector(rank_method="invalid")
    expected = "The 'rank_method' parameter of FOCISelector must be"
    with pytest.raises(InvalidParameterError, match=re.escape(expected)):
        sel.fit(X, y)


def test_fit_raises_when_y_is_none():
    X = np.arange(10.0).reshape(-1, 1)
    sel = FOCISelector()
    with pytest.raises(ValueError, match="y must be provided"):
        sel.fit(X, y=None)


@pytest.mark.parametrize("max_features", [0, -1])
def test_fit_raises_when_max_features_invalid(max_features):
    n, p = 12, 5
    random_state = np.random.RandomState(0)
    X = random_state.normal(size=(n, p))
    y = random_state.normal(size=n)

    sel = FOCISelector(max_features=max_features)
    expected = "The 'max_features' parameter of FOCISelector must be"
    with pytest.raises(InvalidParameterError, match=re.escape(expected)):
        sel.fit(X, y)


def test_random_state_accepts_random_state_instance():
    random_state = np.random.RandomState(0)
    X = random_state.normal(size=(20, 3))
    y = random_state.normal(size=20)

    selector = FOCISelector(random_state=np.random.RandomState(0))
    selector.fit(X, y)

    assert selector.support_mask_.shape == (X.shape[1],)


def test_random_state_int_reproducible():
    random_state = np.random.RandomState(0)
    X = random_state.normal(size=(30, 5))
    y = random_state.normal(size=30)

    selector_1 = FOCISelector(random_state=0).fit(X, y)
    selector_2 = FOCISelector(random_state=0).fit(X, y)

    np.testing.assert_array_equal(
        selector_1.selected_indices_,
        selector_2.selected_indices_,
    )
    assert_allclose(
        selector_1.score_path_,
        selector_2.score_path_,
    )


def test_nn_strategy_grouping_and_radius_are_accepted_and_reproducible():
    X_df, y = make_data(n=200, p=10, seed=0)

    for strategy in ("grouping", "radius"):
        selector_1 = FOCISelector(
            random_state=0, nn_strategy=strategy, min_delta=None, max_features=3
        ).fit(X_df, y)
        selector_2 = FOCISelector(
            random_state=0, nn_strategy=strategy, min_delta=None, max_features=3
        ).fit(X_df, y)

        # Non-trivial selection (avoid early stopping selecting none)
        assert selector_1.support_mask_.sum() > 2

        np.testing.assert_array_equal(
            selector_1.selected_indices_,
            selector_2.selected_indices_,
        )
        assert_allclose(
            selector_1.score_path_,
            selector_2.score_path_,
        )


def test_nn_strategy_invalid_raises():
    random_state = np.random.RandomState(0)
    X = random_state.normal(size=(20, 3))
    y = random_state.normal(size=20)

    sel = FOCISelector(nn_strategy="invalid")
    expected = "The 'nn_strategy' parameter of FOCISelector must be"
    with pytest.raises(InvalidParameterError, match=re.escape(expected)):
        sel.fit(X, y)


@pytest.mark.parametrize(
    "method, expected",
    [
        ("max", np.array([2.0, 2.0, 3.0, 5.0, 5.0])),
        ("average", np.array([1.5, 1.5, 3.0, 4.5, 4.5])),
    ],
)
def test_rank_handles_ties(method, expected):
    y = np.array([1.0, 1.0, 2.0, 3.0, 3.0])

    ranks = _rank(y, method=method)

    assert_allclose(ranks, expected)


def test_rank_restores_original_order():
    y = np.array([3.0, 1.0, 3.0, 2.0, 1.0])

    ranks = _rank(y, method="average")

    assert_allclose(ranks, np.array([4.5, 1.5, 4.5, 3.0, 1.5]))


def test_rank_invalid_method_raises():
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match=re.escape("method must be one of")):
        _rank(y, method="invalid")


def test_nn_grouping_based_no_ties():
    """
    Test _nn_grouping_based on a simple dataset where all pairwise distances
    are distinct (no ties and no identical rows).
    """
    # 1D array where nearest neighbors are unique and unambiguous
    # idx 0 (val 0) -> closest is 10 (idx 1)
    # idx 1 (val 10) -> closest is 12 (idx 2)
    # idx 2 (val 12) -> closest is 10 (idx 1)
    # idx 3 (val 30) -> closest is 12 (idx 2)
    X = np.array([[0], [10], [12], [30]])
    nn_ties = collect_nn_ties(_nn_grouping_based, X)

    expected = [np.array([1]), np.array([2]), np.array([1]), np.array([2])]
    assert len(nn_ties) == len(expected)
    for got, exp in zip(nn_ties, expected):
        np.testing.assert_array_equal(got, exp)

    # Also verify that no point includes itself as a tied neighbor
    for i, ties in enumerate(nn_ties):
        assert i not in set(ties.tolist())


def test_nn_grouping_based_identical_rows_pair():
    """
    Test _nn_grouping_based when there is an exact pair of identical rows.
    Identical rows have distance 0 between each other and must list each other
    as the (only) member of their tie sets.
    """
    # Indices 0 and 2 are identical [1.0, 2.0]
    X = np.array(
        [
            [1.0, 2.0],  # idx 0
            [5.0, 5.0],  # idx 1
            [1.0, 2.0],  # idx 2
            [10.0, 10.0],  # idx 3
        ]
    )
    nn_ties = collect_nn_ties(_nn_grouping_based, X)

    np.testing.assert_array_equal(nn_ties[0], np.array([2]))
    np.testing.assert_array_equal(nn_ties[2], np.array([0]))
    # For idx 3 [10, 10], closest unique row is [5, 5] (idx 1)
    np.testing.assert_array_equal(nn_ties[3], np.array([1]))


def test_nn_grouping_based_identical_rows_multiple():
    """
    Test _nn_grouping_based when there are 3 or more identical rows.

    For samples in the identical group, the tie set is all other members of that
    group. No random tie-breaking happens in _nn_grouping_based.
    """
    # 4 identical rows [7], 1 distinct row [100]
    X = np.array([[7], [7], [7], [7], [100]])
    nn_ties = collect_nn_ties(_nn_grouping_based, X)

    expected = [
        np.array([1, 2, 3]),
        np.array([0, 2, 3]),
        np.array([0, 1, 3]),
        np.array([0, 1, 2]),
        np.array([0, 1, 2, 3]),
    ]
    assert len(nn_ties) == len(expected)
    for got, exp in zip(nn_ties, expected):
        np.testing.assert_array_equal(got, exp)


def test_nn_grouping_based_distance_ties():
    """
    Test _nn_grouping_based when there are distance ties between distinct
    unique rows.
    """
    # In 1D: row 0 (val 0) is equidistant to row 1 (val -3) and row 2 (val 3)
    X_1d = np.array([[0], [-3], [3]])
    nn_1d = collect_nn_ties(_nn_grouping_based, X_1d)

    np.testing.assert_array_equal(np.sort(nn_1d[0]), np.array([1, 2]))
    assert 0 not in set(nn_1d[0].tolist())

    # In 2D: row 0 is at (0, 0), surrounded by 4 points at Euclidean distance 1;
    # row 5 is at (10, 10), surrounded by 4 points at Euclidean distance 1.
    X_2d = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
            [10, 10],
            [11, 10],
            [10, 11],
            [9, 10],
            [10, 9],
        ]
    )
    nn_2d = collect_nn_ties(_nn_grouping_based, X_2d)

    np.testing.assert_array_equal(np.sort(nn_2d[0]), np.array([1, 2, 3, 4]))
    np.testing.assert_array_equal(np.sort(nn_2d[5]), np.array([6, 7, 8, 9]))


def test_nn_grouping_based_combined_ties_and_identical_rows():
    """
    Test _nn_grouping_based when there is a distance tie between two different
    unique rows, where one of the unique rows has multiple identical copies.
    """
    # Row 0 is [0]. Minimal distance is 2, achieved by [2] (indices 1 and 2)
    # and [-2] (index 3).
    # All indices belonging to tied unique rows should be pooled as candidates.
    X = np.array(
        [
            [0],  # idx 0
            [2],  # idx 1
            [2],  # idx 2
            [-2],  # idx 3
        ]
    )
    nn_ties = collect_nn_ties(_nn_grouping_based, X)

    # Candidate indices for row 0 are {1, 2, 3}
    np.testing.assert_array_equal(np.sort(nn_ties[0]), np.array([1, 2, 3]))

    # Since idx 1 and 2 are identical, their tie sets are each other
    np.testing.assert_array_equal(nn_ties[1], np.array([2]))
    np.testing.assert_array_equal(nn_ties[2], np.array([1]))

    # For idx 3 [-2], closest is [0] (idx 0)
    np.testing.assert_array_equal(nn_ties[3], np.array([0]))


def test_nn_grouping_based_deterministic():
    """
    Test that _nn_grouping_based is deterministic and does not depend on a RNG.
    """
    X = np.array(
        [
            [0, 0],
            [1, 1],
            [1, 1],
            [1, 1],
            [-1, -1],
            [0, 0],
            [2, 2],
            [-2, -2],
            [-1, -1],
        ]
    )

    nn_ties_1 = collect_nn_ties(_nn_grouping_based, X)
    nn_ties_2 = collect_nn_ties(_nn_grouping_based, X)

    assert len(nn_ties_1) == len(nn_ties_2)
    for t1, t2 in zip(nn_ties_1, nn_ties_2):
        np.testing.assert_array_equal(np.sort(t1), np.sort(t2))

    for i, ties in enumerate(nn_ties_1):
        assert i not in set(ties.tolist())


def test_nn_grouping_based_all_identical_rows():
    """
    Test behavior when all samples in X are identical.
    Each sample should list all other samples as its tie set.
    """
    n = 10
    X = np.ones((n, 3))
    nn_ties = collect_nn_ties(_nn_grouping_based, X)

    assert len(nn_ties) == n
    for i in range(n):
        np.testing.assert_array_equal(
            nn_ties[i], np.array([j for j in range(n) if j != i])
        )


def test_nn_grouping_based_matches_radius_based_no_ties():
    """
    On random continuous data without ties or identical rows, grouping-based
    and radius-based NN tie sets should match and be singletons.
    """
    random_state = np.random.RandomState(0)
    X = random_state.normal(size=(50, 5))

    nn_grouping = collect_nn_ties(_nn_grouping_based, X)
    nn_radius = collect_nn_ties(_nn_radius_based, X)

    assert len(nn_grouping) == len(nn_radius)
    for g, r in zip(nn_grouping, nn_radius):
        np.testing.assert_array_equal(g, r)
        assert g.shape == (1,)


def test_nn_grouping_based_two_samples():
    """
    Test minimal sample size (n_samples == 2).
    With only 2 samples, index 0's tie set must be [1] and index 1's tie set must
    be [0], regardless of whether they are distinct or identical.
    """
    X_distinct = np.array([[10], [20]])
    X_identical = np.array([[5], [5]])

    nn_distinct = collect_nn_ties(_nn_grouping_based, X_distinct)
    nn_identical = collect_nn_ties(_nn_grouping_based, X_identical)

    np.testing.assert_array_equal(nn_distinct[0], np.array([1]))
    np.testing.assert_array_equal(nn_distinct[1], np.array([0]))
    np.testing.assert_array_equal(nn_identical[0], np.array([1]))
    np.testing.assert_array_equal(nn_identical[1], np.array([0]))


@pytest.mark.parametrize("nn_func", [_nn_grouping_based, _nn_radius_based])
def test_nn_helpers_return_aggregated_values(nn_func):
    """
    NN helpers should return one scalar aggregation result per sample.

    The aggregator receives each sample's nearest-neighbor tie set and returns
    the number of tied nearest neighbors. The helper should collect these
    returned scalar values in an ndarray.
    """
    X = np.array([[0.0], [-1.0], [1.0]])

    aggregated = nn_func(X, lambda nn_ties: float(len(nn_ties)))

    assert aggregated.shape == (X.shape[0],)
    assert aggregated.dtype.kind == "f"

    # sample 0 has two equidistant nearest neighbors; samples 1 and 2 each have
    # one nearest neighbor.
    np.testing.assert_array_equal(aggregated, np.array([2.0, 1.0, 1.0]))


def test_nn_grouping_based_returns_aggregated_values_for_identical_rows():
    """
    _nn_grouping_based should aggregate identical-row tie sets immediately.

    In this dataset:
      - samples 0, 1, and 2 are identical, so each has two tied neighbors
      - sample 3 has the identical-row group as nearest neighbors, so it has
        three tied neighbors
    """
    X = np.array([[1.0], [1.0], [1.0], [5.0]])

    aggregated = _nn_grouping_based(X, lambda nn_ties: float(len(nn_ties)))

    np.testing.assert_array_equal(aggregated, np.array([2.0, 2.0, 2.0, 3.0]))


def test_Tn_invalid_tie_breaking_raises():
    X = np.array([[0.0], [1.0], [2.0]])
    y_rank = np.array([1.0, 2.0, 3.0])
    random_state = np.random.RandomState(0)

    expected = (
        "nn_tie_breaking must be one of {'random', 'mean', 'first'}, got 'invalid'."
    )
    with pytest.raises(ValueError, match=re.escape(expected)):
        _Tn(
            X,
            y_rank,
            random_state,
            nn_strategy="grouping",
            nn_tie_breaking="invalid",
        )


def test_Tn_mean_tie_breaking_grouping():
    """
    Cover nn_tie_breaking="mean" on a dataset with a true distance tie.

    For X = [[0], [-1], [1]]:
      - sample 0 has two nearest neighbors at equal distance: indices {1, 2}
      - using mean tie-breaking sets neighbor rank for sample 0 to mean([2, 3]) = 2.5
    """
    X = np.array([[0.0], [-1.0], [1.0]])
    y_rank = np.array([1.0, 2.0, 3.0])
    random_state = np.random.RandomState(0)

    tn = _Tn(
        X,
        y_rank,
        random_state,
        nn_strategy="grouping",
        nn_tie_breaking="mean",
    )
    assert np.isfinite(tn)


def test_Tn_mean_tie_breaking_radius():
    """
    Cover nn_strategy="radius" together with nn_tie_breaking="mean".
    """
    X = np.array([[0.0], [-1.0], [1.0]])
    y_rank = np.array([1.0, 2.0, 3.0])
    random_state = np.random.RandomState(0)

    tn = _Tn(
        X,
        y_rank,
        random_state,
        nn_strategy="radius",
        nn_tie_breaking="mean",
    )
    assert np.isfinite(tn)


def test_nn_first_based_unique_neighbors():
    """Hand-computed nearest neighbors on data without ties."""
    X = np.array([[0.0], [2.0], [3.0], [100.0]])
    expected = np.array([1, 2, 1, 2])
    np.testing.assert_array_equal(_nn_first_based(X), expected)


def test_nn_first_based_distance_tie():
    """
    On a distance tie, one of the tied neighbors is returned, never self.

    For X = [[0], [-1], [1]], sample 0 has the two tied neighbors {1, 2}.
    """
    X = np.array([[0.0], [-1.0], [1.0]])
    nn_first = _nn_first_based(X)
    assert nn_first[0] in (1, 2)
    assert nn_first[1] == 0
    assert nn_first[2] == 0
    assert np.all(nn_first != np.arange(3))


def test_nn_first_based_identical_rows():
    """With identical rows, a non-self identical sample is returned."""
    X = np.array([[0.0], [0.0], [0.0]])
    nn_first = _nn_first_based(X)
    assert np.all(nn_first != np.arange(3))
    assert set(nn_first.tolist()) <= {0, 1, 2}


def test_nn_first_based_repeatable():
    """Repeated calls return identical neighbors on the same system."""
    X = np.random.RandomState(0).normal(size=(50, 3))
    np.testing.assert_array_equal(_nn_first_based(X), _nn_first_based(X))


def test_Tn_first_tie_breaking_hand_computed():
    """
    Explicit reference value for T_n with nn_tie_breaking="first".

    On tie-free data the neighbor indices are unambiguous, so the result can
    be computed by hand and serves as a regression check. On such data all
    tie-breaking methods must agree.

    For X = [[0], [2], [3]] and y_rank = [1, 2, 3] the nearest-neighbor
    ranks are [2, 3, 2], giving
    T_n = 1 - 3/8 * 3 + 3/8 * (7 + 6 - 12) = 0.25.
    """
    X = np.array([[0.0], [2.0], [3.0]])
    y_rank = np.array([1.0, 2.0, 3.0])
    random_state = np.random.RandomState(0)

    tn = _Tn(
        X,
        y_rank,
        random_state,
        nn_strategy="grouping",
        nn_tie_breaking="first",
    )
    assert tn == pytest.approx(0.25)

    for nn_tie_breaking in ("random", "mean"):
        tn_other = _Tn(
            X,
            y_rank,
            random_state,
            nn_strategy="grouping",
            nn_tie_breaking=nn_tie_breaking,
        )
        assert tn_other == pytest.approx(tn)


def test_Tn_first_tie_breaking_ignores_nn_strategy():
    """nn_strategy is ignored for 'first': both strategies agree, even with ties."""
    rng = np.random.RandomState(0)
    X = rng.randn(30, 2)
    # Create duplicate rows to exercise tie handling in grouping/radius
    X[1] = X[0]
    X[10] = X[9]
    y_rank = rng.permutation(30).astype(float) + 1.0
    random_state = np.random.RandomState(0)

    tn_grouping = _Tn(
        X,
        y_rank,
        random_state,
        nn_strategy="grouping",
        nn_tie_breaking="first",
    )
    tn_radius = _Tn(
        X,
        y_rank,
        random_state,
        nn_strategy="radius",
        nn_tie_breaking="first",
    )
    assert tn_radius == pytest.approx(tn_grouping)


def test_Qn_first_tie_breaking_hand_computed():
    """
    Explicit reference value for Q_n with nn_tie_breaking="first".

    On tie-free data the neighbor indices are unambiguous, allowing a manual
    calculation that serves as a regression check.

    For X = [[0], [2], [3]], y = [1, 2, 3], rank_method="max":
      R = [1, 2, 3], L = [3, 2, 1], neighbor ranks R_nbr = [2, 3, 2].
      Q_n = sum(min(R, R_nbr) - L^2 / n) / n^2 = (5 - 14/3) / 9 = 1/27.
    """
    X = np.array([[0.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0])
    R = _rank(y, method="max")
    L = _rank(-y, method="max")

    qn = _Qn(
        X,
        R,
        L,
        np.random.RandomState(0),
        nn_strategy="grouping",
        nn_tie_breaking="first",
    )
    assert qn == pytest.approx(1.0 / 27.0)


def test_first_tie_breaking_ignores_random_state():
    """nn_tie_breaking="first" results are independent of random_state."""
    X_df, y = make_data(n=100, p=10, seed=0)
    results = []
    for random_state in (0, 123):
        selector = FOCISelector(
            nn_tie_breaking="first",
            random_state=random_state,
            min_delta=None,
            max_features=4,
        ).fit(X_df, y)
        results.append((selector.selected_indices_, selector.score_path_))

    np.testing.assert_array_equal(results[0][0], results[1][0])
    assert_allclose(results[0][1], results[1][1])


def test_Qn_hand_computed_mean_ties():
    """
    Hand-computed test for _Qn with mean tie-breaking.

    For X = [[0], [-1], [1]], y = [1, 2, 3], rank_method="max":
      R = [1, 2, 3], L = [3, 2, 1].
      Sample 0 has equidistant NN {1, 2} with mean rank (2 + 3) / 2 = 2.5.
      Sample 1 has NN {0} with rank 1.
      Sample 2 has NN {0} with rank 1.
    """
    X = np.array([[0.0], [-1.0], [1.0]])
    y = np.array([1.0, 2.0, 3.0])
    R = _rank(y, method="max")
    L = _rank(-y, method="max")
    n = X.shape[0]

    # Target ranks of nearest neighbors
    R_nbr = np.array([2.5, 1.0, 1.0])
    expected_Q = float(np.sum(np.minimum(R, R_nbr) - L**2 / n) / n**2)

    actual_Q = _Qn(
        X,
        R,
        L,
        np.random.RandomState(0),
        nn_strategy="grouping",
        nn_tie_breaking="mean",
    )
    assert actual_Q == pytest.approx(expected_Q, abs=1e-15)


def test_S_y_constant_target_is_zero():
    """Constant target yields L_i = n (for max rank) so S(y) == 0.0."""
    y = np.array([42.0, 42.0, 42.0, 42.0])
    L = _rank(-y, method="max")
    assert _S_y(L) == 0.0


@pytest.mark.parametrize("n", [5, 10])
def test_S_y_uniform_continuous(n):
    """For strictly continuous uniform y, S(y) = (n^2 - 1) / (6 * n^2)."""
    y = np.arange(1, n + 1, dtype=float)
    expected = (n**2 - 1) / (6 * n**2)
    for rank_method in ("max", "average"):
        L = _rank(-y, method=rank_method)
        assert_allclose(_S_y(L), expected)


def test_Qn_invalid_tie_breaking_raises():
    X = np.array([[0.0], [1.0], [2.0]])
    y_rank = np.array([1.0, 2.0, 3.0])
    y_rank_neg = np.array([3.0, 2.0, 1.0])
    random_state = np.random.RandomState(0)

    expected = (
        "nn_tie_breaking must be one of {'random', 'mean', 'first'}, got 'invalid'."
    )
    with pytest.raises(ValueError, match=re.escape(expected)):
        _Qn(
            X,
            y_rank,
            y_rank_neg,
            random_state,
            nn_strategy="grouping",
            nn_tie_breaking="invalid",
        )


@pytest.mark.parametrize("strategy", ["grouping", "radius"])
def test_Qn_divided_by_Sy_equals_Tn_on_continuous_data(strategy):
    """On continuous data, Q_n(y, X) / S(y) == T_n(y, X) to machine precision."""
    X_df, y = make_data(n=300, p=10, seed=0)
    X_sub = X_df.iloc[:, :3].to_numpy()
    R = _rank(y, method="max")
    L = _rank(-y, method="max")
    rng = np.random.RandomState(0)

    tn = _Tn(
        X_sub,
        R,
        rng,
        nn_strategy=strategy,
        nn_tie_breaking="mean",
    )
    qn = _Qn(
        X_sub,
        R,
        L,
        rng,
        nn_strategy=strategy,
        nn_tie_breaking="mean",
    )
    sy = _S_y(L)

    assert_allclose(qn / sy, tn, atol=1e-12)


def test_n_jobs_zero_raises():
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10.0)

    with pytest.raises(InvalidParameterError, match="'n_jobs'"):
        FOCISelector(n_jobs=0).fit(X, y)


def test_n_jobs_parallel_matches_sequential_for_mean_tie_breaking():
    """Parallel candidate scoring preserves deterministic selection results."""
    X, y = make_data(n=80, p=8, seed=42)
    params = dict(
        random_state=0,
        max_features=3,
        min_delta=None,
        nn_tie_breaking="mean",
    )

    sequential = FOCISelector(**params, n_jobs=1).fit(X, y)
    parallel = FOCISelector(**params, n_jobs=2).fit(X, y)

    np.testing.assert_array_equal(
        parallel.selected_indices_, sequential.selected_indices_
    )
    assert_allclose(parallel.score_path_, sequential.score_path_)


def test_n_jobs_random_ties_match_across_worker_counts():
    """A seeded random tie path is identical in sequential and parallel fits."""
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    params = dict(random_state=42, max_features=2, min_delta=None)

    baseline = FOCISelector(**params).fit(X, y)
    for n_jobs in (1, 2, -1):
        selector = FOCISelector(**params, n_jobs=n_jobs).fit(X, y)
        np.testing.assert_array_equal(
            selector.selected_indices_, baseline.selected_indices_
        )
        assert_allclose(selector.score_path_, baseline.score_path_)


def test_score_candidate_uses_its_assigned_random_seed():
    """A candidate worker has its own deterministic random stream."""
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0.0, 1.0, 2.0, 3.0])
    y_rank = _rank(y)
    y_rank_neg = _rank(-y)
    S_y = _S_y(y_rank_neg)
    seed = 42

    j, score = _score_candidate(
        1,
        [0],
        X,
        y_rank,
        y_rank_neg,
        S_y,
        seed,
        nn_strategy="grouping",
        nn_tie_breaking="random",
        method="ct_foci",
    )

    assert j == 1
    expected = _Tn(
        X[:, [0, 1]],
        y_rank,
        np.random.RandomState(seed),
        nn_strategy="grouping",
        nn_tie_breaking="random",
    )
    assert_allclose(score, expected)


def test_score_candidate_r_foci_uses_its_seed():
    """A candidate worker for r_foci has its own deterministic random stream."""
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0.0, 1.0, 2.0, 3.0])
    y_rank = _rank(y)
    y_rank_neg = _rank(-y)
    S_y = _S_y(y_rank_neg)
    seed = 42

    j, score = _score_candidate(
        1,
        [0],
        X,
        y_rank,
        y_rank_neg,
        S_y,
        seed,
        nn_strategy="grouping",
        nn_tie_breaking="random",
        method="r_foci",
    )

    assert j == 1
    expected_qn = _Qn(
        X[:, [0, 1]],
        y_rank,
        y_rank_neg,
        np.random.RandomState(seed),
        nn_strategy="grouping",
        nn_tie_breaking="random",
    )
    assert_allclose(score, expected_qn / S_y)


def test_method_invalid_raises():
    random_state = np.random.RandomState(0)
    X = random_state.normal(size=(20, 3))
    y = random_state.normal(size=20)

    sel = FOCISelector(method="nonsense")
    expected = "The 'method' parameter of FOCISelector must be"
    with pytest.raises(InvalidParameterError, match=re.escape(expected)):
        sel.fit(X, y)


def test_r_foci_is_default():
    X_df, y = make_data(n=100, p=10, seed=0)
    sel_default = FOCISelector(random_state=0).fit(X_df, y)
    sel_r_foci = FOCISelector(method="r_foci", random_state=0).fit(X_df, y)

    np.testing.assert_array_equal(
        sel_default.selected_indices_, sel_r_foci.selected_indices_
    )
    assert_allclose(sel_default.score_path_, sel_r_foci.score_path_)


@pytest.mark.parametrize("rank_method", ["max", "average"])
def test_r_foci_accepts_both_rank_methods(rank_method):
    X_df, y = make_data(n=100, p=10, seed=0)
    selector = FOCISelector(
        method="r_foci",
        rank_method=rank_method,
        min_delta=None,
        max_features=4,
        random_state=0,
    ).fit(X_df, y)
    assert len(selector.selected_indices_) == 4
    assert selector.score_path_.shape == (4,)
    assert np.all(np.isfinite(selector.score_path_))


@pytest.mark.parametrize("strategy", ["grouping", "radius"])
@pytest.mark.parametrize("tie_breaking", ["random", "mean", "first"])
def test_r_foci_accepts_nn_strategies_and_tie_breaking(strategy, tie_breaking):
    X_df, y = make_data(n=200, p=10, seed=0)
    sel1 = FOCISelector(
        method="r_foci",
        nn_strategy=strategy,
        nn_tie_breaking=tie_breaking,
        random_state=0,
        min_delta=None,
        max_features=3,
    ).fit(X_df, y)
    sel2 = FOCISelector(
        method="r_foci",
        nn_strategy=strategy,
        nn_tie_breaking=tie_breaking,
        random_state=0,
        min_delta=None,
        max_features=3,
    ).fit(X_df, y)
    np.testing.assert_array_equal(sel1.selected_indices_, sel2.selected_indices_)
    assert_allclose(sel1.score_path_, sel2.score_path_)


def test_methods_agree_on_continuous_data():
    X_df, y = make_data(n=400, p=20, seed=0)

    sel_rfoci = FOCISelector(
        method="r_foci",
        nn_tie_breaking="mean",
        min_delta=None,
        max_features=5,
        random_state=0,
    ).fit(X_df, y)
    sel_ctfoci = FOCISelector(
        method="ct_foci",
        nn_tie_breaking="mean",
        min_delta=None,
        max_features=5,
        random_state=0,
    ).fit(X_df, y)
    np.testing.assert_array_equal(
        sel_rfoci.selected_indices_, sel_ctfoci.selected_indices_
    )
    assert_allclose(sel_rfoci.score_path_, sel_ctfoci.score_path_, atol=1e-10)

    sel_rfoci_stop = FOCISelector(
        method="r_foci",
        nn_tie_breaking="mean",
        min_delta=0,
        random_state=0,
    ).fit(X_df, y)
    sel_ctfoci_stop = FOCISelector(
        method="ct_foci",
        nn_tie_breaking="mean",
        min_delta=0,
        random_state=0,
    ).fit(X_df, y)
    np.testing.assert_array_equal(
        sel_rfoci_stop.selected_indices_, sel_ctfoci_stop.selected_indices_
    )
    assert_allclose(sel_rfoci_stop.score_path_, sel_ctfoci_stop.score_path_, atol=1e-10)


def test_r_foci_min_delta_zero_may_select_none_on_independent_data():
    random_state = np.random.RandomState(0)
    X = random_state.normal(size=(10, 1))
    y = random_state.normal(size=10)

    selector = FOCISelector(method="r_foci", random_state=0, min_delta=0).fit(X, y)

    assert selector.support_mask_.sum() == 0
    assert len(selector.score_path_) == 0


def test_r_foci_min_delta_none_selects_up_to_max():
    random_state = np.random.RandomState(0)
    X = random_state.normal(size=(20, 5))
    y = random_state.normal(size=20)

    selector = FOCISelector(
        method="r_foci", random_state=0, min_delta=None, max_features=3
    ).fit(X, y)

    assert selector.support_mask_.sum() == 3
    assert len(selector.score_path_) == 3


def test_r_foci_min_delta_enforces_gap():
    X_df, y = make_data(n=200, p=10, seed=0)
    min_delta = 0.03

    selector = FOCISelector(method="r_foci", random_state=0, min_delta=min_delta).fit(
        X_df, y
    )
    scores = selector.score_path_
    assert scores.size > 1

    assert scores[0] > min_delta

    diffs = np.diff(scores)
    assert np.all(diffs > min_delta)


@pytest.mark.parametrize("method", ["r_foci", "ct_foci"])
def test_r_foci_constant_y_selects_none(method):
    X = np.random.RandomState(0).normal(size=(20, 3))
    y = np.ones(20)

    selector = FOCISelector(method=method, random_state=0).fit(X, y)
    assert selector.selected_indices_.size == 0
    assert selector.score_path_.size == 0
    assert selector.support_mask_.sum() == 0


@pytest.mark.parametrize("tie_breaking", ["random", "mean"])
def test_r_foci_parallel_matches_sequential(tie_breaking):
    X, y = make_data(n=80, p=8, seed=42)
    params = dict(
        method="r_foci",
        random_state=42,
        max_features=3,
        min_delta=None,
        nn_tie_breaking=tie_breaking,
    )

    baseline = FOCISelector(**params, n_jobs=1).fit(X, y)
    for n_jobs in (2, -1):
        selector = FOCISelector(**params, n_jobs=n_jobs).fit(X, y)
        np.testing.assert_array_equal(
            selector.selected_indices_, baseline.selected_indices_
        )
        assert_allclose(selector.score_path_, baseline.score_path_)


_RSCRIPT = shutil.which("Rscript")


def test_r_foci_matches_R_reference_skips_when_no_rscript(monkeypatch, tmp_path):
    monkeypatch.setattr("pyFOCI.tests.test_foci._RSCRIPT", None)
    with pytest.raises(pytest.skip.Exception, match="Rscript not available"):
        test_r_foci_matches_R_reference(tmp_path)


def test_r_foci_matches_R_reference_skips_when_package_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("pyFOCI.tests.test_foci._RSCRIPT", "/fake/Rscript")

    def fake_run(*args, **kwargs):
        class FakeProbe:
            stdout = "FALSE\n"

        return FakeProbe()

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(pytest.skip.Exception, match="R package 'FOCI' not installed"):
        test_r_foci_matches_R_reference(tmp_path)


def test_r_foci_matches_R_reference(tmp_path):
    if _RSCRIPT is None:
        pytest.skip("Rscript not available")

    probe = subprocess.run(
        [_RSCRIPT, "-e", 'cat(requireNamespace("FOCI", quietly=TRUE))'],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if probe.stdout.strip() != "TRUE":
        pytest.skip("R package 'FOCI' not installed")

    rng = np.random.RandomState(12345)
    n, p = 300, 12
    X = rng.normal(size=(n, p))
    y = (
        X[:, 0] * X[:, 1]
        + np.sin(X[:, 0] * X[:, 2])
        + X[:, 3] ** 2
        + 0.3 * rng.normal(size=n)
    )
    X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)

    data = np.column_stack([y, X_scaled])
    np.savetxt(tmp_path / "data.csv", data, delimiter=",")

    oracle_r = tmp_path / "oracle.R"
    oracle_r.write_text(
        """data <- read.csv("data.csv", header=FALSE)
Y <- data[, 1]
X <- as.matrix(data[, -1])
set.seed(2024)
res <- FOCI::foci(
  Y, as.matrix(X), stop=TRUE, numCores=1, standardize="none", na.rm=FALSE
)
cat("IDX", paste(res$selectedVar$index, collapse=","), "\n")
cat("T", paste(format(res$stepT, digits=17), collapse=","), "\n")
"""
    )

    out = subprocess.run(
        [_RSCRIPT, str(oracle_r)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )

    idx_line, t_line = "", ""
    for line in out.stdout.splitlines():
        if line.startswith("IDX"):
            idx_line = line.split(maxsplit=1)[1].strip()
        if line.startswith("T"):
            t_line = line.split(maxsplit=1)[1].strip()

    r_indices = [int(x) - 1 for x in idx_line.split(",") if x]
    r_step_t = [float(x) for x in t_line.split(",") if x]

    # Mean tie-breaking comparison (tight tolerance)
    selector_mean = FOCISelector(
        method="r_foci",
        standardize=None,
        min_delta=0,
        nn_strategy="grouping",
        nn_tie_breaking="mean",
        rank_method="max",
        random_state=2024,
    ).fit(X_scaled, y)

    np.testing.assert_array_equal(selector_mean.selected_indices_, r_indices)
    assert_allclose(selector_mean.score_path_, r_step_t, atol=1e-10, rtol=1e-10)

    # Random tie-breaking comparison
    selector_rand = FOCISelector(
        method="r_foci",
        standardize=None,
        min_delta=0,
        nn_strategy="grouping",
        nn_tie_breaking="random",
        rank_method="max",
        random_state=2024,
    ).fit(X_scaled, y)

    np.testing.assert_array_equal(selector_rand.selected_indices_, r_indices)
    assert_allclose(selector_rand.score_path_, r_step_t, atol=1e-9)


def test_r_foci_num_features_equiv():
    """Without R: max_features=k selects k features with non-decreasing scores."""
    X_df, y = make_data(n=300, p=10, seed=0)
    k = 4
    selector = FOCISelector(
        method="r_foci", min_delta=None, max_features=k, random_state=0
    ).fit(X_df, y)

    assert len(selector.selected_indices_) == k
    assert len(selector.score_path_) == k
    diffs = np.diff(selector.score_path_)
    assert np.all(diffs >= -1e-12)
