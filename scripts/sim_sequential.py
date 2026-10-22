"""Simulation study 4: anytime-valid sequential comparison versus fixed-n and naive peeking.

Tasks arrive in random order; each task contributes the paired difference of its R-run means.
Compared: (i) AgentAudit betting confidence sequence (stop at the first decision), (ii) a fixed-n
paired t-test at the full task budget, (iii) naive peeking (a t-test after every task, stop at
p < 0.05). Writes results/sim_sequential.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from agentaudit.sequential import sequential_compare
from agentaudit.simulate import SyntheticPopulation

N_TASKS, RUNS, REPS = 300, 3, 600


def peek_stop(d: np.ndarray, alpha: float = 0.05, min_n: int = 10) -> int | None:
    n = np.arange(1, d.size + 1)
    cm = np.cumsum(d)
    cs = np.cumsum(d**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (cs - cm**2 / n) / (n - 1)
        t = (cm / n) / np.sqrt(var / n)
    p = 2 * stats.t.sf(np.abs(t), np.maximum(n - 1, 1))
    hit = np.flatnonzero((n >= min_n) & (p < alpha))
    return int(hit[0]) + 1 if hit.size else None


def main() -> None:
    rows = []
    for delta_theta in (0.0, 0.2, 0.4, 0.8):
        pop = SyntheticPopulation((0.0, delta_theta), sigma_task=1.2, sigma_interaction=0.5)
        truth = pop.true_diff(1, 0)
        dec = {"seq": 0, "fixed": 0, "peek": 0}
        wrong = {"seq": 0, "peek": 0}
        used: list[int] = []
        for i in range(REPS):
            rs = pop.sample(N_TASKS, RUNS, np.random.default_rng([5, int(delta_theta * 10), i]))
            d = rs.scores[1].mean(axis=1) - rs.scores[0].mean(axis=1)
            res = sequential_compare(d, alpha=0.05)
            if res.decision != "undecided":
                dec["seq"] += 1
                used.append(res.n_used)
                sign_ok = (res.decision == "a_better") == (truth > 0) if truth != 0 else False
                wrong["seq"] += not sign_ok
            sd = d.std(ddof=1)
            if sd > 0:
                dec["fixed"] += (
                    2 * stats.t.sf(abs(d.mean()) / (sd / np.sqrt(N_TASKS)), N_TASKS - 1) < 0.05
                )
            ps = peek_stop(d)
            if ps is not None:
                dec["peek"] += 1
                wrong["peek"] += truth == 0 or (d[:ps].mean() > 0) != (truth > 0)
        rows.append(
            {
                "delta_theta": delta_theta,
                "true_diff": truth,
                "reps": REPS,
                "decision_rate": {k: v / REPS for k, v in dec.items()},
                "wrong_decision_rate": {k: v / REPS for k, v in wrong.items()},
                "mean_tasks_used_when_stopped": float(np.mean(used)) if used else None,
                "median_tasks_used_when_stopped": float(np.median(used)) if used else None,
                "task_budget": N_TASKS,
            }
        )
        print(rows[-1])
    Path("results/sim_sequential.json").write_text(
        json.dumps({"rows": rows, "alpha": 0.05}, indent=1)
    )


if __name__ == "__main__":
    main()
