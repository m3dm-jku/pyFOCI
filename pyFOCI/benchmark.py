"""Measure local FOCISelector candidate-scoring parallelism.

This is intentionally a benchmark script, rather than a test: performance
varies with available CPUs, BLAS threading, and host load.
"""

import argparse
import os
from time import perf_counter

import numpy as np

from pyFOCI import FOCISelector


def _make_data(n_samples, n_features, seed):
    """Create a deterministic nonlinear regression data set."""
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = (
        X[:, 0] * X[:, 1]
        + np.sin(X[:, 2] * X[:, 3])
        + X[:, 4] ** 2
        + 0.1 * rng.normal(size=n_samples)
    )
    return X, y


def _time_fit(X, y, n_jobs, max_features, repeats, method="r_foci"):
    """Return the fastest of repeated fits after one warm-up fit."""
    params = dict(
        method=method,
        max_features=max_features,
        min_delta=None,
        nn_tie_breaking="mean",
        n_jobs=n_jobs,
    )
    FOCISelector(**params).fit(X, y)  # warm up imports, worker creation, and caches

    elapsed = []
    for _ in range(repeats):
        start = perf_counter()
        FOCISelector(**params).fit(X, y)
        elapsed.append(perf_counter() - start)
    return min(elapsed)


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--features", type=int, default=120)
    parser.add_argument("--max-features", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--method",
        type=str,
        choices=["r_foci", "fuchs"],
        default="r_foci",
        help="Selection scoring method: 'r_foci' (default) or 'fuchs'.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        nargs="+",
        default=None,
        help="Worker counts to measure; defaults to powers of two and -1.",
    )
    args = parser.parse_args()

    cpu_count = os.cpu_count()
    if args.n_jobs is None:
        # Benchmark powers of two below the available CPU count. Use joblib's
        # -1 for the final all-available-processors measurement, which also
        # works when the CPU count is unavailable.
        n_jobs = [1]
        if cpu_count is None or cpu_count > 1:
            while cpu_count is not None and 2 * n_jobs[-1] < cpu_count:
                n_jobs.append(2 * n_jobs[-1])
            n_jobs.append(-1)
    else:
        n_jobs = args.n_jobs

    X, y = _make_data(args.samples, args.features, args.seed)
    print(f"os.cpu_count(): {cpu_count}")
    print(
        f"data: {args.samples} samples, {args.features} features; "
        f"method={args.method!r}; "
        f"max_features={args.max_features}; repeats={args.repeats}"
    )
    print(
        "To avoid oversubscription, set OPENBLAS_NUM_THREADS=1 when using "
        "OpenBLAS, or MKL_NUM_THREADS=1 and/or OMP_NUM_THREADS=1 when using "
        "MKL/OpenMP."
    )

    baseline_elapsed = None
    for worker_count in n_jobs:
        elapsed = _time_fit(
            X, y, worker_count, args.max_features, args.repeats, method=args.method
        )
        if baseline_elapsed is None:
            baseline_elapsed = elapsed
        speedup = baseline_elapsed / elapsed
        print(f"n_jobs={worker_count:>3}: {elapsed:.3f}s ({speedup:.2f}x)")


if __name__ == "__main__":  # pragma: no cover
    _main()
