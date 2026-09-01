pyFOCI - Feature Ordering by Conditional Independence
============================================================

[![tests](https://github.com/m3dm-jku/pyFOCI/actions/workflows/python-app.yml/badge.svg)](https://github.com/m3dm-jku/pyFOCI/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/m3dm-jku/pyFOCI/graph/badge.svg?token=L0XPWwoPLw)](https://codecov.io/gh/m3dm-jku/pyFOCI)
[![doc](https://github.com/m3dm-jku/pyFOCI/actions/workflows/deploy-gh-pages.yml/badge.svg)](https://m3dm-jku.github.io/pyFOCI/)

**pyFOCI** provides the feature selection algorithm "Feature Ordering by Conditional Independence" (FOCI), based on a nonlinear generalization of the partial R² statistic. So it can be especially useful in strongly nonlinear data scenarios.

It is based on
* > Mona Azadkia and Sourav Chatterjee. A simple measure of conditional dependence.
  > The Annals of Statistics, 49(6):3070–3102, 2021. [[DOI]](https://doi.org/10.1214/21-AOS2073) [[arXiv]](https://arxiv.org/abs/1910.12327)
* > Sebastian Fuchs. Quantifying directed dependence via dimension reduction.
  > Journal of Multivariate Analysis 201 (2024): 105266. [[DOI]](https://doi.org/10.1016/j.jmva.2023.105266) [[arXiv]](https://arxiv.org/abs/2112.10147)

The Package is [scikit-learn](https://scikit-learn.org) compatible. It is available [on PyPI](https://pypi.org/project/pyFOCI/).

Refer to the documentation (API and example code) at https://m3dm-jku.github.io/pyFOCI/ .

----

This work has been supported by the COMET-K2 Center of the [Linz Center of Mechatronics (LCM)](https://www.lcm.at/),
funded by the Austrian federal government and the federal state of Upper Austria.
