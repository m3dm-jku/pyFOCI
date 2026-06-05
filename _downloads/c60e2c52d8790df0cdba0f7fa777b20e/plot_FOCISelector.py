"""
===============================
FOCISelector: correlation scree
===============================

This example fits the FOCISelector on synthetic data and plots the absolute
feature–target correlations used by the selector. Selected features are
highlighted.
"""

import matplotlib.pyplot as plt
import numpy as np

from pyFOCI import FOCISelector

rng = np.random.default_rng(0)
n, p = 300, 30
X = rng.normal(size=(n, p))

# Synthetic target with a mixture of marginal and interaction effects
y = (
    X[:, 0] * X[:, 1]
    + np.sin(X[:, 0] * X[:, 2])
    + X[:, 3] ** 2
    + 0.1 * rng.normal(size=n)  # small noise
)

selector = FOCISelector(max_features=6)
selector.fit(X, y)

corr = selector.correlation_
support_idx = set(selector.get_support(indices=True))

# Sort features by descending correlation for a scree-style plot
order = np.argsort(-corr)
corr_sorted = corr[order]
labels_sorted = [f"x{i}" for i in order]

# Color selected features differently
colors = ["tab:orange" if i in support_idx else "tab:blue" for i in order]

plt.figure(figsize=(max(10, p * 0.35), 4))
bars = plt.bar(
    range(p),
    corr_sorted,
    color=colors,
    edgecolor="black",
    linewidth=0.5,
)
plt.axvline(
    x=selector.max_features - 0.5,
    color="k",
    linestyle="--",
    alpha=0.6,
    label="max_features cutoff",
)

plt.title("FOCISelector: absolute feature–target correlations")
plt.ylabel("|corr(Xj, y)|")
plt.xlabel("Features (sorted)")

plt.xticks(ticks=range(p), labels=labels_sorted, rotation=90, fontsize=8)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
