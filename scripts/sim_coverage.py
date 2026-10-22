"""Simulation study 1: empirical coverage and width of interval methods for the benchmark score.

Truth is known exactly (quadrature) for each synthetic population. Writes results/sim_coverage.json.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np

from agentaudit.intervals import (
    beta_binomial,
    beta_binomial_hierarchical,
    bootstrap_ci,
    clopper_pearson,
    wilson,
)
from agentaudit.metrics import batch_pass_hat_k, batch_success
from agentaudit.simulate import SyntheticPopulation

POP = SyntheticPopulation((0.3,), sigma_task=1.5, sigma_interaction=0.5)
METHODS = [
    "wilson_pooled",
    "clopper_pearson_pooled",
    "jeffreys_pooled",
    "bootstrap_tasks",
    "bootstrap_tasks_bca",
    "bootstrap_runs_only",
    "bootstrap_two_stage",
    "beta_binomial_profile",
]


def one_cell(args: tuple[int, int, int, int, int]) -> dict[str, object]:
    t, r, reps, n_boot, seed = args
    truth = POP.true_mean(0)
    hits = dict.fromkeys(METHODS, 0)
    widths = dict.fromkeys(METHODS, 0.0)
    for i in range(reps):
        rs = POP.sample(t, r, np.random.default_rng([seed, t, r, i]))
        y = rs.scores[0]
        c, n = int(y.sum()), y.size
        ivs = {
            "wilson_pooled": wilson(c, n),
            "clopper_pearson_pooled": clopper_pearson(c, n),
            "jeffreys_pooled": beta_binomial(c, n),
            "bootstrap_tasks": bootstrap_ci(y, batch_success, n_boot=n_boot, seed=i),
            "bootstrap_tasks_bca": bootstrap_ci(
                y, batch_success, n_boot=n_boot, method="bca", seed=i
            ),
            "bootstrap_runs_only": bootstrap_ci(
                y, batch_success, n_boot=n_boot, scheme="runs", seed=i
            ),
            "bootstrap_two_stage": bootstrap_ci(
                y, batch_success, n_boot=n_boot, scheme="two-stage", seed=i
            ),
            "beta_binomial_profile": beta_binomial_hierarchical(y.sum(axis=1), np.full(t, r)),
        }
        for m, iv in ivs.items():
            hits[m] += iv.contains(truth)
            widths[m] += iv.width
    return {
        "tasks": t,
        "runs": r,
        "reps": reps,
        "truth": truth,
        "coverage": {m: hits[m] / reps for m in METHODS},
        "mean_width": {m: widths[m] / reps for m in METHODS},
    }


def passk_cell(args: tuple[int, int, int, int, int]) -> dict[str, object]:
    """Coverage of the task bootstrap for pass^k (k = runs // 2)."""
    t, r, reps, n_boot, seed = args
    k = max(1, r // 2)
    truth = POP.true_pass_hat_k(0, k)
    hits = 0
    width = 0.0
    for i in range(reps):
        y = POP.sample(t, r, np.random.default_rng([seed, 99, t, r, i])).scores[0]
        iv = bootstrap_ci(y, batch_pass_hat_k(k), n_boot=n_boot, seed=i)
        hits += iv.contains(truth)
        width += iv.width
    return {
        "tasks": t,
        "runs": r,
        "k": k,
        "truth": truth,
        "coverage": hits / reps,
        "mean_width": width / reps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out", default="results/sim_coverage.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    ns = ap.parse_args()
    tasks = (20, 50) if ns.quick else (20, 50, 100, 300)
    runs = (1, 5) if ns.quick else (1, 3, 10)
    cells = [(t, r, ns.reps, ns.n_boot, 20251) for t, r in product(tasks, runs)]
    t0 = time.time()
    with ProcessPoolExecutor(ns.workers) as ex:
        main_res = list(ex.map(one_cell, cells))
        pk = list(ex.map(passk_cell, [c for c in cells if c[1] >= 3]))
    out = {
        "population": {
            "thetas": POP.thetas,
            "sigma_task": POP.sigma_task,
            "sigma_interaction": POP.sigma_interaction,
        },
        "nominal": 0.95,
        "mean_score": main_res,
        "pass_hat_k": pk,
        "seconds": round(time.time() - t0, 1),
    }
    Path(ns.out).parent.mkdir(parents=True, exist_ok=True)
    Path(ns.out).write_text(json.dumps(out, indent=1))
    print(f"done in {out['seconds']}s")


if __name__ == "__main__":
    main()
