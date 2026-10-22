"""Simulation study 2: type-I error and power of paired tests vs. the common pooled-runs shortcut.

Compared procedures for H0: no difference between agents A and B on the task population.
 - pooled_ztest: two-proportion z-test treating all T*R runs as independent (the shortcut).
 - pooled_mcnemar: McNemar exact on run-paired outcomes (run r of A vs run r of B).
 - task_permutation: sign-flip test on per-task mean differences (AgentAudit).
 - task_bootstrap: cluster bootstrap over tasks (AgentAudit).
 - task_mcnemar: exact McNemar on per-task majority outcomes.
Writes results/sim_tests.json.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats

from agentaudit.compare import compare, mcnemar_exact
from agentaudit.simulate import SyntheticPopulation

PROCS = ["pooled_ztest", "pooled_mcnemar", "task_permutation", "task_bootstrap", "task_mcnemar"]


def pooled_z(a: np.ndarray, b: np.ndarray) -> float:
    n = a.size
    pa, pb = a.mean(), b.mean()
    p = (pa + pb) / 2
    se = np.sqrt(2 * p * (1 - p) / n)
    if se == 0:
        return 1.0
    return float(2 * stats.norm.sf(abs(pa - pb) / se))


def cell(args: tuple[float, int, int, int, int, float]) -> dict[str, object]:
    delta_theta, t, r, reps, seed, sigma_int = args
    pop = SyntheticPopulation((0.0, delta_theta), sigma_task=1.5, sigma_interaction=sigma_int)
    truth = pop.true_diff(1, 0)
    rej = dict.fromkeys(PROCS, 0)
    for i in range(reps):
        rs = pop.sample(t, r, np.random.default_rng([seed, t, r, i, int(delta_theta * 100)]))
        a, b = rs.scores[1], rs.scores[0]
        c = compare(a, b, n_boot=500, seed=i)
        runs_mc = mcnemar_exact(int(np.sum((a == 1) & (b == 0))), int(np.sum((b == 1) & (a == 0))))
        for name, p in (
            ("pooled_ztest", pooled_z(a, b)),
            ("pooled_mcnemar", runs_mc),
            ("task_permutation", c.p_permutation),
            ("task_bootstrap", c.p_bootstrap),
            ("task_mcnemar", c.p_mcnemar),
        ):
            rej[name] += p < 0.05
    return {
        "delta_theta": delta_theta,
        "true_diff": truth,
        "tasks": t,
        "runs": r,
        "interaction_sd": sigma_int,
        "reps": reps,
        "reject_rate": {p: rej[p] / reps for p in PROCS},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--out", default="results/sim_tests.json")
    ap.add_argument("--workers", type=int, default=6)
    ns = ap.parse_args()
    cells = [
        (d, t, r, ns.reps, 7, s)
        for d, t, r, s in product((0.0, 0.2, 0.4), (30, 100, 300), (1, 5, 10), (0.5, 1.0))
    ]
    t0 = time.time()
    with ProcessPoolExecutor(ns.workers) as ex:
        res = list(ex.map(cell, cells))
    out = {"alpha": 0.05, "cells": res, "seconds": round(time.time() - t0, 1)}
    Path(ns.out).write_text(json.dumps(out, indent=1))
    print("done", out["seconds"])


if __name__ == "__main__":
    main()
