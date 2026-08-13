"""
===========================
Average vs max rank in FOCI
===========================

When the target contains ties, ``rank_method="max"`` can suffer from two failure
modes (see :ref:`average_vs_max_ranking` in the User Guide for more information):

- **Premature early stopping:** observed here on **Seed 0**.
- **Selection of spurious distractor features:** observed here on **Seed 16**.

This example compares ``FOCISelector(method="r_foci")`` with ``rank_method="max"``
versus ``rank_method="average"`` across 30 seeds of a discretised nonlinear regression
target, and visualizes the internal mechanics behind both failure modes.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from pyFOCI import FOCISelector
from pyFOCI._foci import _rank, _S_y, _score_candidate

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
        method="r_foci",
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
        return idx, 0.0, sel.score_path_

    pred = HistGradientBoostingRegressor(
        max_iter=100, learning_rate=0.1, max_leaf_nodes=15, random_state=0
    )
    pred.fit(X_train[:, idx], y_train)
    r2 = r2_score(y_test, pred.predict(X_test[:, idx]))
    return idx, r2, sel.score_path_


# -- aggregate comparison --------------------------------------------------
r2_m_list, r2_a_list, dr2_list, tags = [], [], [], []
wins_a = wins_m = ties = 0

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
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(13, 4.2))

# Panel 1: Scatter across 30 seeds
colors = {"same": "tab:gray", "avg": "tab:green", "max": "tab:red"}
labels_map = {"same": "same selection", "avg": "average wins", "max": "max wins"}

for tag in ("same", "avg", "max"):
    mask = np.array([t == tag for t in tags])
    if mask.any():
        ax0.scatter(
            r2_m[mask],
            r2_a[mask],
            c=colors[tag],
            s=50,
            alpha=0.9,
            label=labels_map[tag],
            edgecolor="k",
            linewidth=0.4,
        )

lo = min(r2_m.min(), r2_a.min()) - 0.03
hi = max(r2_m.max(), r2_a.max()) + 0.03
ax0.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")
ax0.set_xlabel("Test R$^2$, rank='max'")
ax0.set_ylabel("Test R$^2$, rank='average'")
ax0.set_title(f"Test R$^2$ across {N_SEEDS} seeds")
ax0.set_xlim(lo, hi)
ax0.set_ylim(lo, hi)

ax0.annotate(
    "Seed 0\n(premature stop)",
    xy=(r2_m[0], r2_a[0]),
    xytext=(r2_m[0] + 0.04, r2_a[0] - 0.08),
    arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    fontsize=8,
)
ax0.annotate(
    "Seed 16\n(spurious noise)",
    xy=(r2_m[16], r2_a[16]),
    xytext=(r2_m[16] + 0.04, r2_a[16] - 0.08),
    arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    fontsize=8,
)

ax0.legend(loc="lower right", fontsize=8)
ax0.set_aspect("equal")

# Panel 2: Seed 0 - Premature early stopping mechanics
X, y = make_data(0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=TRAIN_FRACTION, random_state=0
)
sel_m_full = FOCISelector(
    method="r_foci",
    rank_method="max",
    min_delta=None,
    max_features=4,
    nn_tie_breaking="mean",
    random_state=0,
).fit(X_train, y_train)
sel_a_full = FOCISelector(
    method="r_foci",
    rank_method="average",
    min_delta=None,
    max_features=4,
    nn_tie_breaking="mean",
    random_state=0,
).fit(X_train, y_train)

steps = np.arange(1, 5)
labels_feat = [f"+x{j}" for j in sel_a_full.selected_indices_]
ax1.plot(
    steps[:2],
    sel_m_full.score_path_[:2],
    "o-",
    color="tab:red",
    lw=2,
    label="max (stops at step 2)",
)
ax1.plot(
    steps[1:],
    sel_m_full.score_path_[1:],
    "o--",
    color="tab:red",
    alpha=0.4,
    lw=1.5,
    label="max (without stopping)",
)
ax1.plot(
    steps,
    sel_a_full.score_path_,
    "s-",
    color="tab:green",
    lw=2,
    label="average (selects all 4)",
)
ax1.axvline(2, color="tab:red", linestyle=":", alpha=0.6)
ax1.annotate(
    "Gain <= 0\n(early stop)",
    xy=(3, sel_m_full.score_path_[2]),
    xytext=(2.6, 0.44),
    arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.8),
    fontsize=8,
    color="tab:red",
)
ax1.set_xticks(steps)
ax1.set_xticklabels(
    [f"Step {i}\n({l})" for i, l in zip(steps, labels_feat)], fontsize=8
)
ax1.set_xlabel("Forward selection step")
ax1.set_ylabel("Selection score")
ax1.set_title("Seed 0: Premature early stopping")
ax1.legend(loc="lower right", fontsize=8)
ax1.grid(True, linestyle=":", alpha=0.4)

# Panel 3: Seed 16 - Spurious distractor mechanics (Step 2 candidate ranking)
X, y = make_data(16)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=TRAIN_FRACTION, random_state=16
)
X_s = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

top_cand = [1, 15, 3, 10]
scores_max_cand = []
scores_avg_cand = []
for j in top_cand:
    y_r = _rank(y_train, method="max")
    y_rn = _rank(-y_train, method="max")
    sy = _S_y(y_rn)
    _, sm = _score_candidate(
        j,
        [0],
        X_s,
        y_r,
        y_rn,
        sy,
        seed=0,
        nn_strategy="grouping",
        nn_tie_breaking="mean",
        method="r_foci",
    )
    y_r = _rank(y_train, method="average")
    y_rn = _rank(-y_train, method="average")
    sy = _S_y(y_rn)
    _, sa = _score_candidate(
        j,
        [0],
        X_s,
        y_r,
        y_rn,
        sy,
        seed=0,
        nn_strategy="grouping",
        nn_tie_breaking="mean",
        method="r_foci",
    )
    scores_max_cand.append(sm)
    scores_avg_cand.append(sa)

x_pos = np.arange(len(top_cand))
w = 0.35
cand_labels = [f"x{j}\n({'signal' if j < 5 else 'noise'})" for j in top_cand]
ax2.bar(
    x_pos - w / 2,
    scores_max_cand,
    width=w,
    color="tab:red",
    alpha=0.85,
    label="rank='max'",
)
ax2.bar(
    x_pos + w / 2,
    scores_avg_cand,
    width=w,
    color="tab:green",
    alpha=0.85,
    label="rank='average'",
)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(cand_labels, fontsize=8)
ax2.set_xlabel("Top candidates at Step 2 (given x0)")
ax2.set_ylabel("Selection score")
ax2.set_title("Seed 16: Noise distractor vs. signal")
ax2.set_ylim(0.40, 0.49)
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(True, linestyle=":", alpha=0.4)

fig.tight_layout()
plt.show()

# -- numeric summary -------------------------------------------------------
X, y = make_data(DATA_SEED)
X_train, _, y_train, _ = train_test_split(
    X, y, train_size=TRAIN_FRACTION, random_state=DATA_SEED
)

n_train = len(y_train)
rank_sum_max = _rank(y_train, method="max").sum()
rank_sum_avg = _rank(y_train, method="average").sum()

print(f"Seed {DATA_SEED} rank sums:")
print(f"  rank_method='max'     : {rank_sum_max:.0f}")
print(f"  rank_method='average' : {rank_sum_avg:.0f}")

print(f"\nSeeds 0..{N_SEEDS - 1} (method='r_foci'):")
print(f"  identical selections      : {ties}/{N_SEEDS}")
print(f"  average better test R2    : {wins_a}/{N_SEEDS} (seeds 0 and 16)")
print(f"  max better test R2        : {wins_m}/{N_SEEDS}")
print(f"  mean R2, max              : {r2_m.mean():.3f}")
print(f"  mean R2, average          : {r2_a.mean():.3f}")
print(f"  mean R2 difference        : {dr2.mean():+.4f}")
