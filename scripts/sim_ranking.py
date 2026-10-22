"""Simulation study 5: calibration of ranking uncertainty.

Checks (a) coverage of the 95% bootstrap rank intervals for the *population* ranking and
(b) calibration of P(best): among agent-replications assigned probability p of being the best,
how often is the agent truly best? Writes results/sim_ranking.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agentaudit.ranking import rank_uncertainty
from agentaudit.simulate import SyntheticPopulation

THETAS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.9)
REPS = 400


def main() -> None:
    rows = []
    bins = np.linspace(0, 1, 11)
    for t, r in ((50, 3), (150, 3), (400, 3)):
        pop = SyntheticPopulation(THETAS, sigma_task=1.3, sigma_interaction=0.5)
        true_means = np.array([pop.true_mean(i) for i in range(len(THETAS))])
        true_rank = np.argsort(np.argsort(-true_means)) + 1
        cover = np.zeros(len(THETAS))
        pbest_all, best_all = [], []
        for i in range(REPS):
            rs = pop.sample(t, r, np.random.default_rng([11, t, i]))
            ru = rank_uncertainty(rs, n_boot=600, seed=i)
            cover += (ru.rank_lo <= true_rank) & (true_rank <= ru.rank_hi)
            pbest_all += ru.rank_probs[:, 0].tolist()
            best_all += (true_rank == 1).tolist()
        pb, be = np.array(pbest_all), np.array(best_all)
        cal = []
        for lo, hi in zip(bins[:-1], bins[1:], strict=True):
            sel = (pb >= lo) & (pb < hi + (hi == 1.0))
            if sel.sum() >= 20:
                cal.append(
                    {
                        "p_lo": float(lo),
                        "p_hi": float(hi),
                        "n": int(sel.sum()),
                        "mean_pred": float(pb[sel].mean()),
                        "observed": float(be[sel].mean()),
                    }
                )
        rows.append(
            {
                "tasks": t,
                "runs": r,
                "reps": REPS,
                "rank_interval_coverage": (cover / REPS).tolist(),
                "mean_coverage": float(np.mean(cover / REPS)),
                "calibration": cal,
            }
        )
        print(t, rows[-1]["mean_coverage"], [round(c["observed"] - c["mean_pred"], 2) for c in cal])
    Path("results/sim_ranking.json").write_text(
        json.dumps({"thetas": THETAS, "rows": rows}, indent=1)
    )


if __name__ == "__main__":
    main()
