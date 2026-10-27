"""Re-analysis of public tau-bench trajectories (gpt-4o vs claude-3.5-sonnet, 4 trials/task).

Writes results/reanalysis_taubench.json with intervals, pass^k, paired tests (task-level vs the
pooled-runs shortcut), variance decomposition and the power needed to confirm the observed gap.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from agentaudit.analysis import analyze
from agentaudit.compare import compare, holm, mcnemar_exact
from agentaudit.ingest import taubench_resultset
from agentaudit.power import DiffVariance, required_tasks

D = Path("data/raw/taubench")


def pooled_tests(a: np.ndarray, b: np.ndarray) -> dict[str, float]:  # type: ignore[type-arg]
    """The shortcut: treat every run as an independent observation."""
    n = a.size
    pa, pb = a.mean(), b.mean()
    p = (pa + pb) / 2
    z = abs(pa - pb) / np.sqrt(2 * p * (1 - p) / n)
    return {
        "pooled_ztest_p": float(2 * stats.norm.sf(z)),
        "pooled_run_mcnemar_p": mcnemar_exact(
            int(np.sum((a == 1) & (b == 0))), int(np.sum((b == 1) & (a == 0)))
        ),
    }


def needed_interval(a: np.ndarray, b: np.ndarray, n_boot: int = 500) -> dict[str, float]:  # type: ignore[type-arg]
    """Bootstrap (over tasks) distribution of 'tasks needed for 80% power at the observed gap'.

    The pilot variance components and the observed gap are both estimated, so the plug-in figure is uncertain.
    """
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, a.shape[0], a.shape[0])
        aa, bb = a[idx], b[idx]
        diff = float(aa.mean() - bb.mean())
        vals.append(
            required_tasks(max(abs(diff), 0.01), aa.shape[1], DiffVariance.from_pilot(aa, bb))
        )
    q = np.quantile(vals, [0.025, 0.5, 0.975])
    return {"p2.5": float(q[0]), "median": float(q[1]), "p97.5": float(q[2]), "n_boot": n_boot}


def main() -> None:
    out: dict[str, object] = {"source": "sierra-research/tau-bench historical_trajectories (MIT)"}
    for domain in ("airline", "retail"):
        rs = taubench_resultset(
            {
                "gpt-4o": D / f"gpt-4o-{domain}.json",
                "claude-3.5-sonnet": D / f"sonnet-35-new-{domain}.json",
            }
        )
        a = analyze(rs, n_boot=5000, mde_target=0.05)
        sonnet, gpt = rs.model("claude-3.5-sonnet"), rs.model("gpt-4o")
        cmp = compare(sonnet, gpt, names=("claude-3.5-sonnet", "gpt-4o"), n_boot=10000)
        dv = DiffVariance.from_pilot(sonnet, gpt)
        out[domain] = {
            "n_tasks": rs.n_tasks,
            "n_runs": rs.n_runs,
            "analysis": a,
            "task_level_comparison": {
                "diff": cmp.diff,
                "ci": [cmp.ci_lo, cmp.ci_hi],
                "p_permutation": cmp.p_permutation,
                "p_bootstrap": cmp.p_bootstrap,
                "p_task_mcnemar": cmp.p_mcnemar,
                "prob_superiority": cmp.prob_superiority,
            },
            "pooled_shortcut": pooled_tests(sonnet, gpt),
            "tasks_needed_for_80pct_power_at_observed_gap": required_tasks(
                max(abs(cmp.diff), 0.01), rs.n_runs, dv
            )
            if cmp.diff
            else None,
            "pilot_diff_variance": {"task": dv.task, "run": dv.run},
            "tasks_needed_bootstrap_interval": needed_interval(sonnet, gpt),
        }
        print(
            domain,
            f"diff={cmp.diff:+.3f} CI=[{cmp.ci_lo:+.3f},{cmp.ci_hi:+.3f}] "
            f"p_perm={cmp.p_permutation:.4f} "
            f"pooled_z p={out[domain]['pooled_shortcut']['pooled_ztest_p']:.5f}",  # type: ignore[index]
        )
    out["holm_p_permutation_two_domains"] = {  # type: ignore[assignment]
        k: float(v)
        for k, v in zip(
            ("airline", "retail"),
            holm([out[d]["task_level_comparison"]["p_permutation"] for d in ("airline", "retail")]),  # type: ignore[index]
            strict=True,
        )
    }
    Path("results/reanalysis_taubench.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
