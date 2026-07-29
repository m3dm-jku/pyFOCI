"""
===========================
Average vs max rank in FOCI
===========================

The Fuchs (2024) :math:`T_n` formula contains the term

.. math::
   \\sum_{i=1}^n R_{\\mathrm{N}(i)}
   \\;+
   \\sum_{i=1}^n R_i
   \\;-
   n(n+1).

The :math:`-n(n+1)` offset is derived for continuous :math:`y`, where
:math:`\\sum_i R_i = n(n+1)/2`.  With tied :math:`y`, the
``rank_method="max"`` (as used in the papers' proofs)
inflates the rank sum and can push :math:`T_n` above 1.

Average ranking preserves the rank-sum identity with ties.  This example
discretises a nonlinear regression target into 5 equal-frequency bins and
compares feature selection with ``rank_method="max"`` and
``rank_method="average"``.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from pyFOCI import FOCISelector
from pyFOCI._foci import _rank

# -- config ----------------------------------------------------------------
N_SAMPLES = 350
N_FEATURES = 20
N_INFORMATIVE = 5
N_LEVELS = 5
NOISE_SIGMA = 0.5
TRAIN_FRACTION = 0.75
K_FEAT_CAP = 10
DATA_SEED = 0
N_SEEDS = 30


def make_data(seed, n=N_SAMPLES, p=N_FEATURES, n_levels=N_LEVELS, sigma=NOISE_SIGMA):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, p))
    y_lat = (
        2.0 * (X[:, 0] ** 2 - 1.0)
        + 1.5 * np.sin(2.0 * X[:, 1])
        + 2.0 * np.exp(-X[:, 2] ** 2)
        + 1.5 * X[:, 3] * X[:, 4]
        + 1.0 * (X[:, 3] >= 0)
    )
    y_lat += sigma * rng.normal(size=n)
    q = np.quantile(y_lat, np.linspace(0, 1, n_levels + 1))
    q[0] -= 1e-9
    q[-1] += 1e-9
    y = np.digitize(y_lat, q[1:-1]).astype(float)
    return X, y


def evaluate(rank_method, X_train, X_test, y_train, y_test):
    sel = FOCISelector(
        max_features=K_FEAT_CAP,
        min_delta=0,
        nn_tie_breaking="mean",
        nn_strategy="grouping",
        standardize="normalize",
        rank_method=rank_method,
        random_state=0,
    )
    sel.fit(X_train, y_train)

    idx = sel.selected_indices_
    if len(idx) == 0:
        return idx, 0.0, sel.Tn_path_

    pred = HistGradientBoostingRegressor(
        max_iter=100, learning_rate=0.1, max_leaf_nodes=15, random_state=0
    )
    pred.fit(X_train[:, idx], y_train)
    r2 = r2_score(y_test, pred.predict(X_test[:, idx]))
    return idx, r2, sel.Tn_path_


# -- aggregate comparison --------------------------------------------------
r2_m_list, r2_a_list, dr2_list, tags = [], [], [], []
wins_a = wins_m = ties = n_sup1_m = n_sup1_a = 0

for s in range(N_SEEDS):
    X, y = make_data(s)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_FRACTION, random_state=s
    )

    idx_m, r2_m, T_m = evaluate("max", X_train, X_test, y_train, y_test)
    idx_a, r2_a, T_a = evaluate("average", X_train, X_test, y_train, y_test)

    r2_m_list.append(r2_m)
    r2_a_list.append(r2_a)
    dr2_list.append(r2_a - r2_m)

    if max(T_m) > 1 + 1e-9:
        n_sup1_m += 1
    if len(T_a) > 0 and max(T_a) > 1 + 1e-9:
        n_sup1_a += 1

    if list(idx_m) == list(idx_a):
        ties += 1
        tags.append("same")
    elif r2_a - r2_m > 1e-9:
        wins_a += 1
        tags.append("avg")
    elif r2_a - r2_m < -1e-9:
        wins_m += 1
        tags.append("max")
    else:
        ties += 1
        tags.append("same")

r2_m = np.array(r2_m_list)
r2_a = np.array(r2_a_list)
dr2 = np.array(dr2_list)

# -- plot ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 5.0))

colors = {"same": "tab:gray", "avg": "tab:green", "max": "tab:red"}
labels_map = {"same": "same selection", "avg": "average wins", "max": "max wins"}

for tag in ("same", "avg", "max"):
    mask = np.array([t == tag for t in tags])
    if mask.any():
        ax.scatter(
            r2_m[mask],
            r2_a[mask],
            c=colors[tag],
            s=55,
            alpha=0.9,
            label=labels_map[tag],
            edgecolor="k",
            linewidth=0.4,
        )

lo = min(r2_m.min(), r2_a.min()) - 0.03
hi = max(r2_m.max(), r2_a.max()) + 0.03
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")
ax.set_xlabel("Test R$^2$, rank='max'")
ax.set_ylabel("Test R$^2$, rank='average'")
ax.set_title(f"Average vs max ranking across {N_SEEDS} seeds")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.legend(loc="best", fontsize=8)
ax.set_aspect("equal")

fig.tight_layout()

# -- numeric summary -------------------------------------------------------
X, y = make_data(DATA_SEED)
X_train, _, y_train, _ = train_test_split(
    X, y, train_size=TRAIN_FRACTION, random_state=DATA_SEED
)

n_train = len(y_train)
expected_rank_sum = n_train * (n_train + 1) / 2
rank_sum_max = _rank(y_train, method="max").sum()
rank_sum_avg = _rank(y_train, method="average").sum()

print(f"Seed {DATA_SEED} rank sums:")
print(f"  expected              : {expected_rank_sum:.0f}")
print(
    f"  rank_method='max'     : {rank_sum_max:.0f} "
    f"(excess {rank_sum_max - expected_rank_sum:+.0f})"
)
print(
    f"  rank_method='average' : {rank_sum_avg:.0f} "
    f"(excess {rank_sum_avg - expected_rank_sum:+.0f})"
)

print(f"\nSeeds 0..{N_SEEDS - 1}:")
print(f"  identical selections      : {ties}/{N_SEEDS}")
print(f"  average better test R2    : {wins_a}/{N_SEEDS}")
print(f"  max better test R2        : {wins_m}/{N_SEEDS}")
print(f"  seeds with T_n > 1, max   : {n_sup1_m}/{N_SEEDS}")
print(f"  seeds with T_n > 1, avg   : {n_sup1_a}/{N_SEEDS}")
print(f"  mean R2, max              : {r2_m.mean():.3f}")
print(f"  mean R2, average          : {r2_a.mean():.3f}")
print(f"  mean R2 difference        : {dr2.mean():+.4f}")
