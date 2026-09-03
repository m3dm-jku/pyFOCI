Testing and merging:
====================

1. Locally:
-----------

# To update "latest", e.g. on a new Python release:
pixi update

pixi run -e lint lint

# Optional: to also run the black-box cross-check against the CRAN FOCI reference,
# install R (https://cran.r-project.org) and in R: install.packages("FOCI")
# The test self-skips when Rscript or the FOCI package are unavailable.

pixi run -e latest test
# Also examine the resulting coverage.xml.

pixi run -e doc build-doc
# Also examine the resulting example plot.
# Run single examples with:
# pixi run -e doc python examples/filename.py

Commit to a dev branch.

pixi run build
pixi run check-dist

2. CI:
------

Push dev branch to GitHub.

Create pull request to trigger CI tests.

Update the [Unreleased] changelog section, if relevant

# Merge locally, as a workaround for https://github.com/orgs/community/discussions/5524,
# "PR's "Rebase and Merge" should not alter commits if the head branch is already on top of the main one".
printf 'Fast-forwarding main to dev: '; git merge-base --is-ancestor main dev && git branch -f main dev && echo OK || echo ERROR

git push origin main:main
# By this, The GitHub repo setting "Automatically delete head branches"
# deletes the dev branch in the pull request, for easier creation of the next pull request from dev.

Check the formatting of https://github.com/m3dm-jku/pyFOCI/blob/main/CHANGELOG.md


Releasing:
==========

Select a version like `0.1.2`, following [Semantic Versioning](https://semver.org/).

Update the changelog, commit with "Release 0.1.2".

# Tag the commit as "v0.1.2" and push it, including the tag:
git push origin main:main v0.1.2

Check the GitHub and PyPI release pages.

Save changed doc pages to archive.org,
e.g. with https://addons.mozilla.org/en-US/firefox/addon/web-archive/
or directly on https://web.archive.org/save .

