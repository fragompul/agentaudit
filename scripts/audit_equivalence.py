# ruff: noqa: E501, C408
"""Final audit A: operating characteristics of the sequential equivalence procedure.

Per-task paired differences d_t are drawn i.i.d. from a three-point law on {-1, 0, +1} with
P(+1) - P(-1) = delta and P(+1) + P(-1) = 0.3 (a bounded, discrete, non-Gaussian law, so the
guarantee is tested on a distribution unlike the earlier logistic-normal generator). For each
margin and true difference it records the decision rates with Clopper-Pearson 95% intervals,
the time-uniform miss rate of the confidence sequence for the true mean, and stopping times.
Writes results/audit_equivalence.json.   usage: python scripts/audit_equivalence.py [reps]
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

from agentaudit.sequential import confidence_sequence, sequential_compare

N_TASKS, ACTIVE = 300, 0.3
MARGINS = (0.0, 0.03, 0.06, 0.10)


def cp(k: int, n: int) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(stats.beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def cell(args: tuple[float, float, int]) -> dict:
    margin, delta, reps = args
    p_plus, p_minus = (ACTIVE + delta) / 2, (ACTIVE - delta) / 2
    probs = [p_minus, 1 - ACTIVE, p_plus]
    rng = np.random.default_rng([2026, int(margin * 1000), int(round(delta * 1000)) + 1000])
    cnt = {"a_better": 0, "b_better": 0, "equivalent": 0, "undecided": 0}
    miss = 0
    used: list[int] = []
    for _ in range(reps):
        d = rng.choice([-1.0, 0.0, 1.0], p=probs, size=N_TASKS)
        res = sequential_compare(d, margin=margin)
        cnt[res.decision] += 1
        if res.decision != "undecided":
            used.append(res.n_used)
        lo, hi = confidence_sequence((d + 1) / 2)
        lo, hi = 2 * lo - 1, 2 * hi - 1
        miss += bool(np.any((lo > delta + 1e-12) | (hi < delta - 1e-12)))
    inside = abs(delta) <= margin + 1e-12
    wrong = (
        (cnt["a_better"] if delta <= margin else 0)
        + (cnt["b_better"] if delta >= -margin else 0)
        + (cnt["equivalent"] if not inside else 0)
    )
    return {
        "margin": margin,
        "delta": delta,
        "reps": reps,
        "in_equivalence_region": inside,
        "counts": cnt,
        "rates": {k: v / reps for k, v in cnt.items()},
        "wrong_decision": wrong,
        "wrong_decision_rate": wrong / reps,
        "wrong_decision_cp95": cp(wrong, reps),
        "false_equivalence_cp95": cp(cnt["equivalent"], reps) if not inside else None,
        "cs_miss_rate": miss / reps,
        "cs_miss_cp95": cp(miss, reps),
        "median_tasks_when_decided": float(np.median(used)) if used else None,
    }


def main() -> None:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    cells = []
    for m in MARGINS:
        for delta in sorted({0.0, m, m + 0.02, 2 * m + 0.02, -(m + 0.02)}):
            cells.append((m, delta, reps))
    with ProcessPoolExecutor(12) as ex:
        out = list(ex.map(cell, cells))
    Path("results/audit_equivalence.json").write_text(
        json.dumps(
            {"alpha": 0.05, "n_tasks": N_TASKS, "active_prob": ACTIVE, "cells": out}, indent=1
        )
    )
    for c in out:
        print(
            c["margin"],
            c["delta"],
            c["rates"],
            "wrong",
            c["wrong_decision"],
            "miss",
            c["cs_miss_rate"],
        )


if __name__ == "__main__":
    main()
