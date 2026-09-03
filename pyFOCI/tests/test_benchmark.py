"""Tests for the local benchmark module."""

import sys

import numpy as np

from pyFOCI import benchmark


def test_make_data_is_deterministic_and_has_expected_shape():
    X_first, y_first = benchmark._make_data(n_samples=10, n_features=5, seed=42)
    X_second, y_second = benchmark._make_data(n_samples=10, n_features=5, seed=42)

    assert X_first.shape == (10, 5)
    assert y_first.shape == (10,)
    np.testing.assert_array_equal(X_first, X_second)
    np.testing.assert_array_equal(y_first, y_second)


def test_time_fit_warms_up_and_returns_fastest_repeat(monkeypatch):
    fits = []

    class DummySelector:
        def __init__(self, **params):
            self.params = params

        def fit(self, X, y):
            fits.append((self.params, X, y))
            return self

    clock = iter([0.0, 2.0, 3.0, 8.0])
    monkeypatch.setattr(benchmark, "FOCISelector", DummySelector)
    monkeypatch.setattr(benchmark, "perf_counter", lambda: next(clock))

    X = np.ones((3, 5))
    y = np.ones(3)
    elapsed = benchmark._time_fit(X, y, n_jobs=2, max_features=4, repeats=2)

    assert elapsed == 2.0  # the minimum elapsed time
    assert len(fits) == 3  # one warm-up and two timed fits

    # Parameters were forwarded correctly:
    assert all(params["method"] == "r_foci" for params, _, _ in fits)
    assert all(params["n_jobs"] == 2 for params, _, _ in fits)
    assert all(params["max_features"] == 4 for params, _, _ in fits)
    assert all(params["min_delta"] is None for params, _, _ in fits)
    assert all(params["nn_tie_breaking"] == "mean" for params, _, _ in fits)


def test_time_fit_method_forwarding(monkeypatch):
    fits = []

    class DummySelector:
        def __init__(self, **params):
            self.params = params

        def fit(self, X, y):
            fits.append(self.params)
            return self

    monkeypatch.setattr(benchmark, "FOCISelector", DummySelector)
    monkeypatch.setattr(benchmark, "perf_counter", lambda: 0.0)

    X = np.ones((3, 5))
    y = np.ones(3)
    # Use a non-default method, so the assertion really checks the forwarding:
    benchmark._time_fit(X, y, n_jobs=1, max_features=2, repeats=1, method="ct_foci")
    assert all(params["method"] == "ct_foci" for params in fits)


def test_main_uses_power_of_two_worker_counts(monkeypatch, capsys):
    worker_counts = []
    monkeypatch.setattr(sys, "argv", ["benchmark"])
    monkeypatch.setattr(benchmark.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(benchmark, "_make_data", lambda *args: ("X", "y"))

    def time_fit(X, y, n_jobs, max_features, repeats, method="r_foci"):
        worker_counts.append((X, y, n_jobs, max_features, repeats, method))
        return {1: 4.0, 2: 2.0, -1: 1.0}[n_jobs]

    monkeypatch.setattr(benchmark, "_time_fit", time_fit)

    benchmark._main()

    # powers of two below cpu_count, and -1:
    assert [entry[2] for entry in worker_counts] == [1, 2, -1]
    assert all(entry[5] == "r_foci" for entry in worker_counts)
    # Additionally check the default baseline:
    assert "n_jobs=  1: 4.000s (1.00x)" in capsys.readouterr().out


def test_main_handles_one_or_unknown_cpu_and_explicit_worker_counts(
    monkeypatch, capsys
):
    worker_counts = []
    monkeypatch.setattr(benchmark, "_make_data", lambda *args: ("X", "y"))

    def time_fit(X, y, n_jobs, max_features, repeats, method="r_foci"):
        worker_counts.append(n_jobs)
        return 1.0

    monkeypatch.setattr(benchmark, "_time_fit", time_fit)

    monkeypatch.setattr(sys, "argv", ["benchmark"])
    monkeypatch.setattr(benchmark.os, "cpu_count", lambda: 1)
    benchmark._main()
    assert worker_counts == [1]

    worker_counts.clear()
    monkeypatch.setattr(sys, "argv", ["benchmark"])
    monkeypatch.setattr(benchmark.os, "cpu_count", lambda: None)
    benchmark._main()
    assert worker_counts == [1, -1]

    worker_counts.clear()
    monkeypatch.setattr(sys, "argv", ["benchmark", "--n-jobs", "3", "5"])
    monkeypatch.setattr(benchmark.os, "cpu_count", lambda: 8)
    benchmark._main()
    assert worker_counts == [3, 5]
    # Check that the first cound is used as relative baseline:
    assert "n_jobs=  3: 1.000s (1.00x)" in capsys.readouterr().out


def test_main_uses_method_argument(monkeypatch, capsys):
    methods_passed = []
    monkeypatch.setattr(benchmark, "_make_data", lambda *args: ("X", "y"))

    def time_fit(X, y, n_jobs, max_features, repeats, method="r_foci"):
        methods_passed.append(method)
        return 1.0

    monkeypatch.setattr(benchmark, "_time_fit", time_fit)
    monkeypatch.setattr(benchmark.os, "cpu_count", lambda: 1)

    # Use a non-default method, so the assertion really checks the argument:
    monkeypatch.setattr(sys, "argv", ["benchmark", "--method", "ct_foci"])
    benchmark._main()
    assert methods_passed == ["ct_foci"]
    assert "method='ct_foci'" in capsys.readouterr().out
