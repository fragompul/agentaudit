# ruff: noqa: E501
"""Final audit: coverage of the beta-binomial profile interval when the beta mixing assumption fails.

Task success probabilities come from (a) a Beta law, (b) a two-point mixture {0.05, 0.85}, (c) a
point mass at the mean (no heterogeneity). T=50 tasks, R=10 runs, 500 replications per scenario;
the task bootstrap is the reference. Writes results/audit_bbmix.json.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from agentaudit.intervals import beta_binomial_hierarchical, bootstrap_ci

T, R, REPS = 50, 10, 500


def draw(kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "beta":
        return rng.beta(0.6, 1.4, T)  # mean 0.3
    if kind == "two_point":
        return np.where(rng.random(T) < 0.3125, 0.85, 0.05)  # mean 0.3
    return np.full(T, 0.3)


def cell(kind: str) -> dict:
    rng = np.random.default_rng([77, len(kind)])
    hit_bb = hit_bs = 0
    for _ in range(REPS):
        p = draw(kind, rng)
        c = rng.binomial(R, p).astype(float)
        truth = 0.3
        ci = beta_binomial_hierarchical(c, np.full(T, float(R)))
        hit_bb += ci.lo <= truth <= ci.hi
        y = (rng.random((T, R)) < p[:, None]).astype(float)
        b = bootstrap_ci(y, lambda a: a.mean(axis=(-2, -1)), n_boot=500, seed=rng)
        hit_bs += b.lo <= truth <= b.hi
    se = lambda x: float(np.sqrt(x / REPS * (1 - x / REPS) / REPS))  # noqa: E731
    return {
        "mixing": kind,
        "reps": REPS,
        "beta_binomial_profile_coverage": hit_bb / REPS,
        "beta_binomial_mc_se": se(hit_bb),
        "task_bootstrap_coverage": hit_bs / REPS,
        "task_bootstrap_mc_se": se(hit_bs),
    }


def main() -> None:
    kinds = ["beta", "two_point", "point_mass"]
    with ProcessPoolExecutor(3) as ex:
        out = list(ex.map(cell, kinds))
    Path("results/audit_bbmix.json").write_text(json.dumps({"cells": out}, indent=1))
    for c in out:
        print(c)


if __name__ == "__main__":
    main()
