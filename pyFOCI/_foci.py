"""
Feature Ordering by Conditional Independence (FOCI)
"""

# Authors: Robert Pollak <robert.pollak@jku.at>
# License: BSD 3 clause

from numbers import Real

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, _fit_context
from sklearn.feature_selection import SelectorMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import (
    Integral,
    Interval,
    InvalidParameterError,
    StrOptions,
)
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import validate_data


def _rank(y, method="max"):
    """Compute 1-based ranks with configurable tie handling.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Values to rank.
    method : {"max", "average"}, default="max"
        Method used to assign ranks to tied values.
        If "max", tied values receive the maximum rank in their tie group.
        If "average", tied values receive the average rank in their tie group.

    Returns
    -------
    ranks : ndarray of shape (n_samples,), dtype=float
        One-based ranks.
    """
    if method not in ("max", "average"):
        raise ValueError("method must be one of {'max', 'average'}, got {method!r}.")

    y = np.asarray(y)
    n = y.shape[0]
    idx = np.argsort(y, kind="mergesort")
    y_sorted = y[idx]
    ranks = np.empty(n, dtype=float)
    i = 0

    while i < n:
        j = i
        while j + 1 < n and y_sorted[j + 1] == y_sorted[i]:
            j += 1

        if method == "max":
            rank = j + 1
        else:
            # method == "average"
            rank = 0.5 * (i + j) + 1

        ranks[idx[i : j + 1]] = rank
        i = j + 1

    return ranks


def _nn_radius_based(X_sub, aggregator):
    """Radius-based NN tie set aggregation.

    For each sample i, computes the indices of all samples at the minimal
    distance from i (within a tiny epsilon), excluding i itself, and stores the
    value returned by ``aggregator``.

    Parameters
    ----------
    X_sub : array-like of shape (n_samples, n_selected_features)
    aggregator : callable
        Called as ``aggregator(nn_ties)`` for each sample, where ``nn_ties`` is
        a 1d ndarray[int] containing all indices tied at the minimal
        nearest-neighbor distance from the current sample.

    Returns
    -------
    aggregated : ndarray of shape (n_samples,), dtype=float
        Aggregated nearest-neighbor values returned by ``aggregator``.
    """
    X_sub = np.asarray(X_sub)
    n = X_sub.shape[0]
    aggregated = np.empty(n, dtype=float)

    # Fit NN on X_sub
    nbrs = NearestNeighbors(n_neighbors=2)
    nbrs.fit(X_sub)

    # Get min distances
    distances, _ = nbrs.kneighbors(n_neighbors=1)
    min_distance = distances[:, 0]

    eps = 1e-13  # to get all neighbors of min distance

    # For each i, collect all min-dist neighbors, remove self, and aggregate.
    for i in range(n):
        # Query neighbors in the tight radius around the nearest neighbor distance
        nn_ties = nbrs.radius_neighbors(
            X_sub[i, :].reshape(1, -1), min_distance[i] + eps, return_distance=False
        )[0]
        # Remove self index if present
        nn_ties = nn_ties[nn_ties != i]
        aggregated[i] = aggregator(nn_ties)

    return aggregated


def _nn_first_based(X_sub):
    """First nearest-neighbor selection without tie handling.

    For each sample, return the index of a single nearest neighbor,
    excluding the sample itself. If several neighbors are tied at the
    minimal distance, whichever one the underlying nearest-neighbor method
    returns first is used. No tie set is computed.

    Parameters
    ----------
    X_sub : array-like of shape (n_samples, n_selected_features)
        Sample subset used to compute nearest neighbors.

    Returns
    -------
    nn_first : ndarray of shape (n_samples,), dtype=int
        Index of the first nearest neighbor returned for each sample.
    """
    X_sub = np.asarray(X_sub)
    n = X_sub.shape[0]

    # Query the two nearest points. The query point itself is at distance 0
    # and is usually the first returned index; in that case the second entry
    # is a nearest neighbor. If the first returned index is not the query
    # point (possible when at least two other points also have distance 0),
    # both returned indices are tied nearest neighbors at distance 0 and the
    # first one is used.
    nbrs = NearestNeighbors(n_neighbors=2)
    nbrs.fit(X_sub)
    nn_idx = nbrs.kneighbors(return_distance=False)  # shape (n, 2)

    first = nn_idx[:, 0]
    return np.where(first == np.arange(n), nn_idx[:, 1], first)


def _tied_min_distance_neighbors(group_idx, Xu, groups):
    """Return sample indices from all unique rows tied at minimal distance.

    Excludes the self unique row at `group_idx`.
    """
    diff = Xu - Xu[group_idx]  # (m, p)
    d2_all = (diff * diff).sum(axis=1, dtype=float)  # squared distances
    d2_all[group_idx] = np.inf  # exclude self

    dmin2 = d2_all.min()
    tied_group_idx = np.flatnonzero(d2_all == dmin2)  # tied unique rows
    return np.concatenate([groups[u] for u in tied_group_idx])


def _nn_grouping_based(X_sub, aggregator):
    """Grouping-based NN tie set aggregation.

    For each sample i:
      - If there are other samples with identical X_sub rows (distance 0), the
        tie set is the other members of that identical-row group.
      - Otherwise, the tie set consists of all samples whose unique-row
        representation is tied at the minimal distance.

    The computed tie set is passed to ``aggregator``, and the
    returned value is stored in the result array.

    Precondition: n_samples >= 2.

    Parameters
    ----------
    X_sub : array-like of shape (n_samples, n_selected_features)
    aggregator : callable
        Called as ``aggregator(nn_ties)`` for each sample, where ``nn_ties`` is
        a 1d ndarray[int] containing all indices tied at the minimal
        nearest-neighbor distance from the current sample.

    Returns
    -------
    aggregated : ndarray of shape (n_samples,), dtype=float
        Aggregated nearest-neighbor values returned by ``aggregator``.
    """
    X_sub = np.asarray(X_sub)
    n = X_sub.shape[0]
    aggregated = np.empty(n, dtype=float)

    # 1) Group exactly identical rows
    Xu, inv = np.unique(X_sub, axis=0, return_inverse=True)  # Xu: (m, p)
    # inv maps each sample to its "identical-row group" index in [0, m)
    m = Xu.shape[0]

    groups = [[] for _ in range(m)]
    for sample_idx, group_idx in enumerate(inv):
        groups[group_idx].append(sample_idx)
    groups = [np.asarray(g, dtype=int) for g in groups]

    # ---- Case A: only one unique row (all rows identical) ----
    #
    # In this case, all distances are 0 and every "nearest neighbor" tie is global.
    # Tie set for each i is all indices except i itself.
    if m == 1:
        group = groups[0]  # all indices, size n>=2
        for sample_idx in range(n):
            aggregated[sample_idx] = aggregator(group[group != sample_idx])
        return aggregated

    # NN structure on unique rows
    nn = NearestNeighbors()
    nn.fit(Xu)

    # For m==2: only one non-self neighbor exists for each unique row
    # For m>=3: ask for the 2 nearest (non-self) unique rows to detect distance ties
    n_neighbors = 1 if m == 2 else 2
    nn_dist, nn_idx = nn.kneighbors(n_neighbors=n_neighbors, return_distance=True)

    # ---- Cases B/C: two or more unique rows ----
    for sample_idx in range(n):
        group_idx = inv[sample_idx]
        group = groups[group_idx]

        if group.size > 1:
            aggregated[sample_idx] = aggregator(group[group != sample_idx])
            continue

        unique_nn = (m == 2) or (nn_dist[group_idx, 1] > nn_dist[group_idx, 0])

        if unique_nn:
            nn_group_idx = int(nn_idx[group_idx, 0])
            nn_ties = groups[nn_group_idx]
        else:
            # tie at the minimum distance -> brute-force only for this group_idx
            nn_ties = _tied_min_distance_neighbors(group_idx, Xu, groups)

        aggregated[sample_idx] = aggregator(nn_ties)

    return aggregated


def _nearest_neighbor_y_rank(
    X_sub,
    y_rank,
    random_state,
    *,
    nn_strategy="grouping",
    nn_tie_breaking="random",
):
    """Return nearest-neighbor target ranks for T_n and Q_n statistics.

    Both statistics use the same nearest-neighbor tie handling and differ only
    in how the resulting neighbor ranks enter their formulas. This helper keeps
    the shared tie-breaking and neighbor-strategy dispatch in one place.

    With ``nn_tie_breaking="first"`` a single nearest neighbor is queried per
    sample and no tie set is computed; ``nn_strategy`` and ``random_state`` are
    ignored in this case.
    """
    if nn_tie_breaking not in ("random", "mean", "first"):
        raise ValueError(
            "nn_tie_breaking must be one of {'random', 'mean', 'first'}, "
            f"got {nn_tie_breaking!r}."
        )

    if nn_tie_breaking == "first":
        # A single nearest neighbor per sample; no tie set is computed.
        return y_rank[_nn_first_based(X_sub)]

    if nn_tie_breaking == "random":

        def aggregate_nn_ties(nn_ties):
            return float(y_rank[nn_ties[random_state.randint(len(nn_ties))]])

    else:
        # nn_tie_breaking == "mean"
        def aggregate_nn_ties(nn_ties):
            arr = y_rank[nn_ties]
            return float(arr.sum() / len(arr))

    # Neighbor target ranks are kept as float to support mean tie-breaking and
    # average target ranks.
    if nn_strategy == "grouping":
        return _nn_grouping_based(X_sub, aggregate_nn_ties)

    assert nn_strategy == "radius"
    return _nn_radius_based(X_sub, aggregate_nn_ties)


def _Tn(
    X_sub,
    y_rank,
    random_state,
    *,
    nn_strategy="grouping",
    nn_tie_breaking="random",
):
    """Compute :math:`T_n` following Fuchs (2024).

    The implementation uses the expression for :math:`T_n` given in
    Section 4.2 after "straightforward calculation" in:

        Fuchs, Sebastian. "Quantifying directed dependence via dimension
        reduction." Journal of Multivariate Analysis 201 (2024): 105266.

    Parameters
    ----------
    X_sub : array-like of shape (n_samples, n_selected_features)
        Candidate subset of the input features used to compute nearest
        neighbors.
    y_rank : ndarray of shape (n_samples,)
        One-based ranks of the target values.
    random_state : numpy.random.RandomState
        Random number generator used to break nearest-neighbor ties. Only
        used for ``nn_tie_breaking="random"``.
    nn_strategy : {"grouping", "radius"}, default="grouping"
        Strategy used to compute nearest neighbor tie sets. Ignored when
        ``nn_tie_breaking="first"``.
    nn_tie_breaking : {"random", "mean", "first"}, default="random"
        How to resolve ties among equally-distanced nearest neighbors.
        If "random", one tied neighbor is selected at random.
        If "mean", the mean ``y_rank`` across all tied neighbors is used.
        If "first", a single nearest neighbor per sample is queried; the first
        one returned by the nearest-neighbor method is used without computing
        tie sets.

    Returns
    -------
    Tn : float
        Value of the :math:`T_n` statistic for ``X_sub`` and ``y_rank``.
    """
    X_sub = np.asarray(X_sub)
    y_rank = np.asarray(y_rank, dtype=float)
    n = X_sub.shape[0]
    y_rank_nbr = _nearest_neighbor_y_rank(
        X_sub,
        y_rank,
        random_state,
        nn_strategy=nn_strategy,
        nn_tie_breaking=nn_tie_breaking,
    )

    # Apply the formula (indices are 0-based; y_rank is 1-based)
    term1 = np.sum(np.abs(y_rank - y_rank_nbr))
    term2 = np.sum(y_rank_nbr) + np.sum(y_rank) - n * (n + 1)
    result = 1 - 3 / (n**2 - 1) * term1 + 3 / (n**2 - 1) * term2
    return float(result)


def _Qn(
    X_sub,
    y_rank,
    y_rank_neg,
    random_state,
    *,
    nn_strategy="grouping",
    nn_tie_breaking="random",
):
    """Q_n(y, X_sub) = (1/n^2) sum_i [min(R_i, R_{N(i)}) - L_i^2 / n].

    Numerator of the Azadkia--Chatterjee unconditional conditional-dependence
    coefficient. ``y_rank`` are the 1-based ranks R_i of y and ``y_rank_neg`` are
    the 1-based ranks L_i of -y, both computed with :func:`_rank`.

    Parameters
    ----------
    X_sub : array-like of shape (n_samples, n_selected_features)
        Candidate subset of the input features used to compute nearest
        neighbors.
    y_rank : ndarray of shape (n_samples,)
        One-based ranks of the target values.
    y_rank_neg : ndarray of shape (n_samples,)
        One-based ranks of the negated target values.
    random_state : numpy.random.RandomState
        Random number generator used to break nearest-neighbor ties. Only
        used for ``nn_tie_breaking="random"``.
    nn_strategy : {"grouping", "radius"}, default="grouping"
        Strategy used to compute nearest neighbor tie sets. Ignored when
        ``nn_tie_breaking="first"``.
    nn_tie_breaking : {"random", "mean", "first"}, default="random"
        How to resolve ties among equally-distanced nearest neighbors.
        If "random", one tied neighbor is selected at random.
        If "mean", the mean ``y_rank`` across all tied neighbors is used.
        If "first", a single nearest neighbor per sample is queried; the first
        one returned by the nearest-neighbor method is used without computing
        tie sets.

    Returns
    -------
    Qn : float
        Value of the Q_n statistic for ``X_sub``, ``y_rank``, and ``y_rank_neg``.
    """
    X_sub = np.asarray(X_sub)
    y_rank = np.asarray(y_rank, dtype=float)
    y_rank_neg = np.asarray(y_rank_neg, dtype=float)
    n = X_sub.shape[0]
    y_rank_nbr = _nearest_neighbor_y_rank(
        X_sub,
        y_rank,
        random_state,
        nn_strategy=nn_strategy,
        nn_tie_breaking=nn_tie_breaking,
    )

    Q = np.sum(np.minimum(y_rank, y_rank_nbr) - y_rank_neg**2 / n) / n**2
    return float(Q)


def _S_y(y_rank_neg):
    """S(y) = (1/n^3) sum_i L_i (n - L_i), constant in the features."""
    L = np.asarray(y_rank_neg, dtype=float)
    n = L.shape[0]
    return float(np.sum(L * (n - L)) / n**3)


def _score_candidate(
    j,
    selected,
    X,
    y_rank,
    y_rank_neg,
    S_y,
    seed,
    nn_strategy,
    nn_tie_breaking,
    method="r_foci",
):
    """Return the selection score for adding feature ``j`` to ``selected``.

    ``seed`` gives each parallel candidate evaluation an independent random
    stream, avoiding shared mutable random state between worker processes.
    """
    X_sub = X[:, selected + [j]]
    rng = np.random.RandomState(seed)
    if method == "r_foci":
        qn = _Qn(
            X_sub,
            y_rank,
            y_rank_neg,
            rng,
            nn_strategy=nn_strategy,
            nn_tie_breaking=nn_tie_breaking,
        )
        score = qn / S_y if S_y > 0 else 1.0
    else:
        assert method == "fuchs"
        score = _Tn(
            X_sub,
            y_rank,
            rng,
            nn_strategy=nn_strategy,
            nn_tie_breaking=nn_tie_breaking,
        )
    return j, score


class FOCISelector(SelectorMixin, BaseEstimator):
    """
    Feature selector using hierarchical forward selection based on the
    nonlinear Azadkia–Chatterjee T_n coefficient, using the FOCI R reference
    form by default and the Fuchs form as an alternative (see references).

    At each step, among remaining features, we choose the feature that maximizes
    the per-step score on the growing set S_k = S_{k-1} ∪ {j}.

    Parameters
    ----------
    max_features : int or None, default=None
        Maximum number of features to select. If None, no hard cap is applied
        and selection proceeds until early stopping (if `min_delta` is not None)
        or until all features are selected (if `min_delta` is None).

    min_delta : float or None, default=0
        Minimum required improvement in the selection score to continue selecting.
        Behavior:

          - First step:
            select a feature only if best_score > min_delta; otherwise, select none.
          - Subsequent steps:
            continue only if best_score > previous_best + min_delta; otherwise, stop.
          - None disables early stopping (select up to `max_features`).

        Notes:

          - min_delta can be negative to relax stopping,
            0 to reproduce standard early stopping,
            and positive to require stricter improvement.

        Compatibility with the reference implementation:

          - min_delta == 0 corresponds to stop=TRUE
          - min_delta is None corresponds to stop=FALSE

    method : {"r_foci", "fuchs"}, default="r_foci"
        Selection scoring method:

        - "r_foci" (default): Azadkia–Chatterjee :math:`Q_n/S(y)`
          numerator/denominator form, matching the FOCI R reference
          implementation's selection and stopping.
        - "fuchs": the form derived by Fuchs (2024) for continuous targets.

    standardize : {"normalize", None}, default="normalize"
        If "normalize", each column of X is standardized to zero mean and unit
        variance before computing nearest neighbors. If None, X is used as-is.
        Columns with zero variance are left unchanged.

    rank_method : {"max", "average"}, default="max"
        Method used to assign one-based ranks to the target values.
        If "max", tied values receive the maximum rank in their tie group.
        If "average", tied values receive the average rank in their tie group.

    nn_strategy : {"grouping", "radius"}, default="grouping"
        Strategy used to compute nearest neighbor tie sets. Ignored when
        ``nn_tie_breaking="first"``.

    nn_tie_breaking : {"random", "mean", "first"}, default="random"
        How to resolve ties among equally-distanced nearest neighbors.
        If "random", one tied neighbor is selected at random.
        If "mean", the mean target rank across all tied neighbors is used.
        If "first", a single nearest neighbor per sample is queried; the first
        one returned by the nearest-neighbor method is used without computing
        tie sets. This option is available to study its impact on the outcome,
        stability and speed of the algorithm.

    random_state : int, RandomState instance or None, default=None
        Controls the random tie-breaking among nearest neighbors.
        Only used for ``nn_tie_breaking="random"``. Pass an int
        for reproducible results across multiple calls. If None, the global
        NumPy random state is used.

    n_jobs : int or None, default=None
        Number of parallel worker processes used to score candidate features in
        each forward-selection round. ``None`` and ``1`` score candidates
        sequentially. ``-1`` uses all available processors; values below
        ``-1`` follow joblib's convention. A fixed integer ``random_state``
        assigns deterministic random streams per candidate, so results are
        reproducible and identical across worker counts.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (``n_features_in_``,)
        Feature names seen during fit. Defined only when X has feature names.

    support_mask_ : ndarray of shape (``n_features_in_``,), dtype=bool
        Boolean mask of selected features determined during fit.

    score_path_ : ndarray of shape (n_selected,)
        Values of the per-step selection scores; their interpretation
        depends on ``method``.

    References
    ----------
    Mona Azadkia and Sourav Chatterjee. A simple measure of conditional dependence.
    The Annals of Statistics, 49(6):3070–3102, 2021. https://doi.org/10.1214/21-AOS2073

    R FOCI package (reference implementation): https://cran.r-project.org/package=FOCI

    Sebastian Fuchs. Quantifying directed dependence via dimension reduction.
    Journal of Multivariate Analysis 201 (2024): 105266. https://doi.org/10.1016/j.jmva.2023.105266
    """

    _parameter_constraints = {
        "max_features": [None, Interval(Integral, 1, None, closed="left")],
        "min_delta": [None, Interval(Real, None, None, closed="neither")],
        "method": [StrOptions({"r_foci", "fuchs"})],
        "standardize": [None, StrOptions({"normalize"})],
        "rank_method": [StrOptions({"max", "average"})],
        "nn_strategy": [StrOptions({"grouping", "radius"})],
        "nn_tie_breaking": [StrOptions({"random", "mean", "first"})],
        "random_state": ["random_state"],
        "n_jobs": [
            None,
            Interval(Integral, None, -1, closed="right"),
            Interval(Integral, 1, None, closed="left"),
        ],
    }

    def __init__(
        self,
        max_features=None,
        min_delta=0,
        *,
        method="r_foci",
        standardize="normalize",
        rank_method="max",
        nn_strategy="grouping",
        nn_tie_breaking="random",
        random_state=None,
        n_jobs=None,
    ):
        self.max_features = max_features
        self.min_delta = min_delta
        self.method = method
        self.standardize = standardize
        self.rank_method = rank_method
        self.nn_strategy = nn_strategy
        self.nn_tie_breaking = nn_tie_breaking
        self.random_state = random_state
        self.n_jobs = n_jobs

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """
        Fit the selector by hierarchical forward selection maximizing the
        per-step score (see ``method``) over the growing feature set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        if y is None:
            raise ValueError("y must be provided for feature selection.")
        type_of_target(y, input_name="y", raise_unknown=True)

        X, y = validate_data(
            self, X, y, accept_sparse=False, y_numeric=True
        )  # asserts finite values

        # Standardization (if requested)
        if self.standardize == "normalize":
            X_mean = np.mean(X, axis=0, dtype=float)
            X_std = np.std(X, axis=0, dtype=float, ddof=0)
            # Avoid division by zero: leave zero-variance columns unchanged
            safe_std = X_std.copy()
            safe_std[safe_std == 0] = 1.0
            X = (X - X_mean) / safe_std

        n_samples, n_features = X.shape
        if n_samples < 2:
            raise InvalidParameterError(
                "Just one sample provided. Need at least two for nearest neighbors."
            )

        y_rank = _rank(y, method=self.rank_method)
        y_rank_neg = _rank(-y, method=self.rank_method)
        S_y = _S_y(y_rank_neg)

        if S_y == 0.0:
            self.selected_indices_ = np.asarray([], dtype=int)
            self.score_path_ = np.asarray([], dtype=float)
            mask = np.zeros(n_features, dtype=bool)
            self.support_mask_ = mask
            return self

        random_state = check_random_state(self.random_state)

        max_features = n_features if self.max_features is None else self.max_features

        selected = []  # S_k
        score_path = []
        remaining = list(range(n_features))
        score_prev = -np.inf

        # Forward selection up to max_features
        while remaining and (len(selected) < max_features):
            best_j = None
            best_score = -np.inf

            # Generate seeds in feature order before scoring candidates. Each
            # candidate therefore receives the same random stream whether it is
            # evaluated sequentially or in a joblib worker process.
            seeds = random_state.randint(
                np.iinfo(np.uint32).max, size=len(remaining), dtype=np.uint32
            )
            score_args = (
                (
                    j,
                    selected,
                    X,
                    y_rank,
                    y_rank_neg,
                    S_y,
                    int(seed),
                    self.nn_strategy,
                    self.nn_tie_breaking,
                    self.method,
                )
                for j, seed in zip(remaining, seeds)
            )
            if self.n_jobs is None or self.n_jobs == 1:
                scores = [_score_candidate(*args) for args in score_args]
            else:
                scores = Parallel(n_jobs=self.n_jobs)(
                    delayed(_score_candidate)(*args) for args in score_args
                )

            # On equal scores retain the first feature in ``remaining``. This
            # is achieved with lexicographic sorting: first by score, then by
            # smallest index.
            best_j, best_score = max(scores, key=lambda score: (score[1], -score[0]))

            # Early stopping behavior controlled by self.min_delta
            if self.min_delta is not None:
                # First step: if best_score <= 0 + min_delta, select nothing and return
                if len(selected) == 0 and best_score <= 0 + self.min_delta:
                    self.selected_indices_ = np.asarray([], dtype=int)
                    self.score_path_ = np.asarray([], dtype=float)
                    mask = np.zeros(n_features, dtype=bool)
                    self.support_mask_ = mask
                    return self
                # Subsequent steps: stop if no sufficient improvement
                if len(selected) > 0 and best_score <= score_prev + self.min_delta:
                    break

            # Always add the best feature this round
            selected.append(best_j)
            score_path.append(best_score)
            remaining.remove(best_j)
            score_prev = best_score

        # Persist learned attributes
        self.selected_indices_ = np.asarray(selected, dtype=int)
        self.score_path_ = np.asarray(score_path, dtype=float)

        # Build mask
        mask = np.zeros(n_features, dtype=bool)
        mask[self.selected_indices_] = True
        self.support_mask_ = mask

        return self

    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected.
        """
        # SelectorMixin will call this during transform/get_support
        return self.support_mask_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False
        return tags
