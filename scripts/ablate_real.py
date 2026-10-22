"""Ablations on real data: how tasks, runs and the stopping rule change conclusions.

For a directory of trajectories (see run_real.py) the full data is the best available reference and
we ask, for sub-designs with T' tasks and R' runs per task:

  * rank reliability: Kendall tau between the sub-design ranking and the full-data ranking;
  * how many agent pairs are declared distinguishable (paired t-test on task means, Holm);
  * sign agreement with the full-data order for pairs that are significant in the full data;
  * sequential stopping: confidence-sequence decisions and tasks used versus the fixed-n test.

Writes results/ablation_<condition>.json.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

from agentaudit.compare import holm
from agentaudit.sequential import sequential_compare


def pair_pvalues(tm: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    """Paired t-test p-value for every model pair; ``tm`` is ``(models, tasks)``."""
    m = tm.shape[0]
    out = []
    for i, j in combinations(range(m), 2):
        d = tm[i] - tm[j]
        if np.allclose(d, d[0]):
            out.append(1.0 if d[0] == 0 else 0.0)
        else:
            out.append(float(stats.ttest_1samp(d, 0.0).pvalue))
    return np.array(out)


def main() -> None:
    from analyze_real import load  # same directory

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/real/main")
    ap.add_argument("--min-runs", type=int, default=10)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--out", default=None)
    ns = ap.parse_args()
    _, rs = load(Path(ns.dir), ns.min_runs)
    x = rs.scores  # (M, T, R)
    m, t, r = x.shape
    full_tm = x.mean(axis=2)
    full_rank = full_tm.mean(axis=1)
    full_p = holm(pair_pvalues(full_tm))
    pairs = list(combinations(range(m), 2))
    full_sign = np.array([np.sign(full_rank[i] - full_rank[j]) for i, j in pairs])
    full_sig = full_p < 0.05
    rng = np.random.default_rng(0)
    grid = []
    t_grid = sorted({max(10, int(round(t * f))) for f in (0.1, 0.2, 0.4, 0.7, 1.0)})
    r_grid = sorted({k for k in (1, 2, 3, 5, 10, r) if k <= r})
    for tp in t_grid:
        for rp in r_grid:
            taus, ndist, agree = [], [], []
            for _ in range(ns.draws):
                ti = rng.choice(t, size=tp, replace=False)
                ri = rng.choice(r, size=rp, replace=False)
                sub = x[:, ti][:, :, ri].mean(axis=2)
                score = sub.mean(axis=1)
                tau = stats.kendalltau(score, full_rank).statistic
                taus.append(0.0 if np.isnan(tau) else tau)
                p = holm(pair_pvalues(sub))
                ndist.append(float(np.mean(p < 0.05)))
                sg = np.array([np.sign(score[i] - score[j]) for i, j in pairs])
                if full_sig.any():
                    agree.append(float(np.mean(sg[full_sig] == full_sign[full_sig])))
            grid.append(
                {
                    "tasks": tp,
                    "runs": rp,
                    "tau_mean": float(np.mean(taus)),
                    "tau_p05": float(np.quantile(taus, 0.05)),
                    "frac_pairs_distinguishable": float(np.mean(ndist)),
                    "sign_agreement_on_full_significant": float(np.mean(agree)) if agree else None,
                }
            )
    seq = []
    for k, (i, j) in enumerate(pairs):
        d_all = full_tm[i] - full_tm[j]
        dec = {"a_better": 0, "b_better": 0, "equivalent": 0, "undecided": 0}
        used = []
        for _ in range(ns.draws):
            order = rng.permutation(t)
            res = sequential_compare(d_all[order], alpha=0.05)
            dec[res.decision] += 1
            if res.decision != "undecided":
                used.append(res.n_used)
        seq.append(
            {
                "a": rs.models[i],
                "b": rs.models[j],
                "full_data_significant_holm": bool(full_sig[k]),
                "mean_diff": float(d_all.mean()),
                "decisions": {a: v / ns.draws for a, v in dec.items()},
                "median_tasks_when_decided": float(np.median(used)) if used else None,
                "n_tasks": t,
            }
        )
    out = {
        "models": list(rs.models),
        "full": {"n_tasks": t, "n_runs": r, "frac_pairs_distinguishable": float(full_sig.mean())},
        "subdesign": grid,
        "sequential": seq,
    }
    dest = Path(ns.out or f"results/ablation_{Path(ns.dir).name}.json")
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    main()
