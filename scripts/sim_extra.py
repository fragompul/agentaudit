"""Additional simulation experiments added in the scientific revision (pre-specified in docs/experiments.md).

  asym    E1  type-I error of paired tests when d_t is skewed with mean zero (weak null), not symmetric
  alloc   E2  tasks versus repeats across task heterogeneity, interaction, cost ratios and a capped task pool
  covrob  E3  interval coverage across prevalence, task heterogeneity, T and R
  seqm    E5  sequential stopping: margin / min-tasks sensitivity and false-equivalence rate
  rankpw  E6  pairwise distinguishability and Holm family-wise error on a leaderboard with near ties

Run ``python scripts/sim_extra.py <exp>``; each writes results/sim_<exp>.json. All streams are seeded.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from agentaudit.compare import (
    holm,
    mcnemar_exact,
    paired_bootstrap,
    sign_flip_permutation,
    studentized_sign_flip,
)
from agentaudit.intervals import beta_binomial_hierarchical, bootstrap_ci, wilson
from agentaudit.metrics import batch_success
from agentaudit.power import DiffVariance, optimal_runs, power
from agentaudit.sequential import sequential_compare
from agentaudit.simulate import SyntheticPopulation
from agentaudit.variance import decompose_array

WORKERS = 14


def _pmap(fn: Any, cells: list[Any]) -> list[Any]:
    with ProcessPoolExecutor(WORKERS) as ex:
        return list(ex.map(fn, cells))


def _mc_se(p: float, n: int) -> float:
    return float(np.sqrt(max(p * (1 - p), 1e-12) / n))


def _t_p(d: np.ndarray, t: int) -> bool:
    sd = d.std(ddof=1)
    return bool(sd > 0 and 2 * stats.t.sf(abs(d.mean()) / (sd / np.sqrt(t)), t - 1) < 0.05)


# ---------------------------------------------------------------- E1 asymmetric null
def _asym_cell(args: tuple[int, int, float, int]) -> dict[str, Any]:
    t, r, q, reps = args
    rej = dict.fromkeys(("sign_flip", "studentized_sign_flip", "paired_bootstrap", "paired_t"), 0)
    rng = np.random.default_rng([31, t, r, int(q * 100)])
    up = 0.45
    down = up * q / (1 - q)  # E[delta] = q*up - (1-q)*down = 0
    for _ in range(reps):
        u = rng.uniform(0.3, 0.5, t)
        delta = np.where(rng.random(t) < q, up, -down)
        a = rng.binomial(r, u + delta) / r
        b = rng.binomial(r, u) / r
        d = a - b
        rej["sign_flip"] += sign_flip_permutation(d, n_perm=2000, seed=1) < 0.05
        rej["studentized_sign_flip"] += studentized_sign_flip(d, n_perm=2000, seed=1) < 0.05
        rej["paired_bootstrap"] += paired_bootstrap(d, n_boot=1000, seed=1)[2] < 0.05
        rej["paired_t"] += _t_p(d, t)
    rates = {k: v / reps for k, v in rej.items()}
    skew = float(stats.skew(np.where(rng.random(200000) < q, up, -down)))
    return {"tasks": t, "runs": r, "q_big_win": q, "reps": reps, "reject_rate": rates,
            "mc_se_max": max(_mc_se(v, reps) for v in rates.values()), "skewness_of_delta": skew}


def exp_asym() -> dict[str, Any]:
    cells = [(t, r, q, 1500) for t, r, q in product((20, 50, 100, 300), (1, 5), (0.5, 0.1, 0.03))]
    return {"alpha": 0.05, "note": "q=0.5 is the symmetric control", "cells": _pmap(_asym_cell, cells)}


# ---------------------------------------------------------------- E2 allocation
def _alloc_cell(args: tuple[float, float, int]) -> dict[str, Any]:
    s_task, s_int, reps = args
    pop = SyntheticPopulation((0.0, 0.25), sigma_task=s_task, sigma_interaction=s_int)
    big = pop.sample(3000, 30, np.random.default_rng(5))
    dv = DiffVariance.from_components(decompose_array(big.scores))
    delta = pop.true_diff(1, 0)
    b = 1200
    rows = []
    for r in (1, 2, 5, 10, 20):
        t = b // r
        rng = np.random.default_rng([77, int(s_task * 10), int(s_int * 10), r])
        diffs, rej = [], 0
        for _ in range(reps):
            rs = pop.sample(t, r, rng)
            d = rs.scores[1].mean(axis=1) - rs.scores[0].mean(axis=1)
            diffs.append(d.mean())
            rej += _t_p(d, t)
        rows.append({"runs": r, "tasks": t, "empirical_se": float(np.std(diffs)),
                     "analytic_se": float(np.sqrt(dv.var_mean(t, r))), "power_sim": rej / reps,
                     "power_analytic": power(delta, t, r, dv)})
    capped = [{"runs": r, "power_analytic": power(delta, 120, r, dv)} for r in (1, 2, 5, 10, 20, 50)]
    rstar = {str(c): optimal_runs(dv, cost_task=c, cost_run=1.0) for c in (0.0, 1.0, 10.0, 100.0)}
    return {"sigma_task": s_task, "sigma_interaction": s_int, "true_diff": delta,
            "s_t": dv.task, "s_e": dv.run, "budget": b, "fixed_budget": rows,
            "task_pool_capped_at_120": capped, "optimal_runs_by_cost_ratio": rstar, "reps": reps}


def exp_alloc() -> dict[str, Any]:
    return {"cells": _pmap(_alloc_cell, [(a, b, 1500) for a, b in product((0.5, 1.5, 2.5), (0.2, 0.6, 1.0))])}


# ---------------------------------------------------------------- E3 coverage robustness
def _cov_cell(args: tuple[float, float, int, int, int]) -> dict[str, Any]:
    theta, s_task, t, r, reps = args
    pop = SyntheticPopulation((theta,), sigma_task=s_task, sigma_interaction=0.5)
    truth = pop.true_mean(0)
    hit: dict[str, int] = {}
    for i in range(reps):
        seed = [91, int((theta + 5) * 10), int(s_task * 10), t, r, i]
        y = pop.sample(t, r, np.random.default_rng(seed)).scores[0]
        ivs = {
            "wilson_pooled": wilson(int(y.sum()), y.size),
            "bootstrap_tasks": bootstrap_ci(y, batch_success, n_boot=500, seed=i),
            "bootstrap_tasks_bca": bootstrap_ci(y, batch_success, n_boot=500, method="bca", seed=i),
            "bootstrap_runs_only": bootstrap_ci(y, batch_success, n_boot=500, scheme="runs", seed=i),
            "bootstrap_two_stage": bootstrap_ci(y, batch_success, n_boot=500, scheme="two-stage", seed=i),
            "beta_binomial_profile": beta_binomial_hierarchical(y.sum(axis=1), np.full(t, r)),
        }
        for k, iv in ivs.items():
            hit[k] = hit.get(k, 0) + iv.contains(truth)
    return {"theta": theta, "true_mean": truth, "sigma_task": s_task, "tasks": t, "runs": r,
            "reps": reps, "coverage": {k: v / reps for k, v in hit.items()},
            "mc_se_at_95": _mc_se(0.95, reps)}


def exp_covrob() -> dict[str, Any]:
    cells = [(th, s, t, r, 1000) for th, s, t, r in
             product((-2.5, 0.3, 2.5), (0.3, 1.5), (20, 50, 100), (1, 3, 10))]
    return {"nominal": 0.95, "cells": _pmap(_cov_cell, cells)}


# ---------------------------------------------------------------- E5 sequential
def _seq_cell(args: tuple[float, float, int, int]) -> dict[str, Any]:
    dth, margin, min_tasks, reps = args
    pop = SyntheticPopulation((0.0, dth), sigma_task=1.2, sigma_interaction=0.5)
    truth = pop.true_diff(1, 0)
    cnt = {"a_better": 0, "b_better": 0, "equivalent": 0, "undecided": 0}
    used = []
    for i in range(reps):
        rs = pop.sample(300, 3, np.random.default_rng([53, int(dth * 100), i]))
        d = rs.scores[1].mean(axis=1) - rs.scores[0].mean(axis=1)
        res = sequential_compare(d, margin=margin, min_tasks=min_tasks)
        cnt[res.decision] += 1
        if res.decision != "undecided":
            used.append(res.n_used)
    return {"delta_theta": dth, "true_diff": truth, "margin": margin, "min_tasks": min_tasks, "reps": reps,
            "rates": {k: v / reps for k, v in cnt.items()},
            "median_tasks_when_decided": float(np.median(used)) if used else None}


def exp_seqm() -> dict[str, Any]:
    cells = [(d, m, 10, 500) for d, m in product((0.0, 0.1, 0.2, 0.4), (0.0, 0.03, 0.06, 0.10))]
    cells += [(d, 0.0, 30, 500) for d in (0.0, 0.2)]
    return {"alpha": 0.05, "budget_tasks": 300, "runs": 3, "cells": _pmap(_seq_cell, cells)}


# ---------------------------------------------------------------- E6 ranking pairs
def _rank_cell(args: tuple[int, int, int]) -> dict[str, Any]:
    t, r, reps = args
    thetas = (0.0, 0.0, 0.15, 0.3, 0.6, 1.0)
    pop = SyntheticPopulation(thetas, sigma_task=1.3, sigma_interaction=0.5)
    pairs = list(combinations(range(len(thetas)), 2))
    truth = np.array([abs(pop.true_diff(i, j)) for i, j in pairs])
    rej = np.zeros(len(pairs))
    fw = 0
    for i in range(reps):
        rs = pop.sample(t, r, np.random.default_rng([61, t, r, i]))
        ps = []
        for a, b in pairs:
            x, y = rs.scores[a], rs.scores[b]
            if r == 1:
                ps.append(mcnemar_exact(int(np.sum((x == 1) & (y == 0))), int(np.sum((y == 1) & (x == 0)))))
            else:
                d = x.mean(axis=1) - y.mean(axis=1)
                sd = d.std(ddof=1)
                ps.append(1.0 if sd == 0 else float(2 * stats.t.sf(abs(d.mean()) / (sd / np.sqrt(t)), t - 1)))
        sig = holm(np.array(ps)) < 0.05
        rej += sig
        fw += bool(sig[0])  # pair (0, 1) has equal true skill
    return {"tasks": t, "runs": r, "reps": reps,
            "pairs": [{"i": i, "j": j, "true_abs_diff": float(g), "holm_reject_rate": float(x / reps)}
                      for (i, j), g, x in zip(pairs, truth, rej, strict=True)],
            "false_rejection_rate_equal_pair": fw / reps}


def exp_rankpw() -> dict[str, Any]:
    cells = [(t, r, 400) for t, r in product((50, 150, 500), (1, 3))]
    return {"thetas": [0.0, 0.0, 0.15, 0.3, 0.6, 1.0], "cells": _pmap(_rank_cell, cells)}


EXPS = {"asym": exp_asym, "alloc": exp_alloc, "covrob": exp_covrob, "seqm": exp_seqm, "rankpw": exp_rankpw}

if __name__ == "__main__":
    name = sys.argv[1]
    t0 = time.time()
    out = EXPS[name]()
    out["seconds"] = round(time.time() - t0, 1)
    Path(f"results/sim_{name}.json").write_text(json.dumps(out, indent=1))
    print(name, "done in", out["seconds"], "s")
