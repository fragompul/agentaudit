"""Simulation study 3: tasks versus repeats at a fixed attempt budget.

For two synthetic agents and a fixed number of attempts B = T * R, compare the empirical standard
error of the estimated score difference and the power of the paired test across R, against the
analytic prediction Var = (s_t + s_e / R) / T from the variance decomposition. Writes
results/sim_design.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from agentaudit.power import DiffVariance, power
from agentaudit.simulate import SyntheticPopulation
from agentaudit.variance import decompose_array

POP = SyntheticPopulation((0.0, 0.25), sigma_task=1.5, sigma_interaction=0.6)
BUDGETS = (240, 600, 1200)
RUNS = (1, 2, 3, 5, 10, 20)
REPS = 3000


def one(b: int, r: int, seed: int) -> dict[str, float]:
    t = b // r
    rng = np.random.default_rng(seed)
    diffs, rejects = [], 0
    for _ in range(REPS):
        rs = POP.sample(t, r, rng)
        d = rs.scores[1].mean(axis=1) - rs.scores[0].mean(axis=1)
        diffs.append(d.mean())
        sd = d.std(ddof=1)
        if sd > 0:
            rejects += 2 * stats.t.sf(abs(d.mean()) / (sd / np.sqrt(t)), t - 1) < 0.05
    return {
        "budget": b,
        "runs": r,
        "tasks": t,
        "empirical_se": float(np.std(diffs)),
        "power": rejects / REPS,
    }


def main() -> None:
    # variance components of the pair from one very large sample (population truth proxy)
    big = POP.sample(4000, 30, np.random.default_rng(123))
    comps = decompose_array(big.scores)
    dv = DiffVariance.from_components(comps)
    delta = POP.true_diff(1, 0)
    rows = []
    for b in BUDGETS:
        for r in RUNS:
            if b // r < 5:
                continue
            row = one(b, r, 1000 * b + r)
            row["analytic_se"] = float(np.sqrt(dv.var_mean(b // r, r)))
            row["analytic_power"] = power(delta, b // r, r, dv)
            rows.append(row)
    out = {
        "population": {
            "thetas": POP.thetas,
            "sigma_task": POP.sigma_task,
            "sigma_interaction": POP.sigma_interaction,
        },
        "true_diff": delta,
        "diff_variance": {"task": dv.task, "run": dv.run},
        "rows": rows,
    }
    Path("results/sim_design.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        if r["budget"] == 1200:
            print(
                r["runs"],
                r["tasks"],
                round(r["empirical_se"], 4),
                round(r["analytic_se"], 4),
                round(r["power"], 3),
                round(r["analytic_power"], 3),
            )


if __name__ == "__main__":
    main()
