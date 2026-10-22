"""One-call statistical audit of a ``ResultSet``: the entry point used by CLI, reports and web."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from agentaudit.compare import pairwise
from agentaudit.intervals import bootstrap_ci
from agentaudit.metrics import batch_pass_at_k, batch_pass_hat_k, batch_success
from agentaudit.power import DiffVariance, mde, optimal_runs, plan, required_tasks
from agentaudit.ranking import rank_uncertainty, split_half_rank_correlation
from agentaudit.results import ResultSet
from agentaudit.variance import decompose_array, decompose_single


@dataclass(frozen=True)
class ModelSummary:
    model: str
    score: float
    lo: float
    hi: float
    pass_hat_k: dict[int, tuple[float, float, float]]
    pass_at_k: dict[int, tuple[float, float, float]]
    task_icc: float | None


def _ks(n_runs: int) -> list[int]:
    return sorted({k for k in (1, 2, 3, 5, 10) if k <= n_runs} | {n_runs})


def analyze(
    results: ResultSet,
    *,
    level: float = 0.95,
    n_boot: int = 2000,
    seed: int = 0,
    mde_target: float = 0.05,
) -> dict[str, Any]:
    """Compute intervals, pass@k/pass^k, variance decomposition, ranking and power planning.

    Returns a JSON-serialisable dictionary. Missing attempts (NaN) are tolerated for scores and
    intervals; variance decomposition, pass^k and pairwise tests use balanced data, so for
    unbalanced inputs they are computed on the tasks/runs observed for every agent.
    """
    rs = results
    balanced = rs.is_balanced()
    if not balanced:
        # runs are exchangeable: keep the first r_ok observed values of every cell, and only
        # the tasks for which every agent has at least r_ok observations
        counts = np.sum(~np.isnan(rs.scores), axis=2)
        runs_ok = max(int(np.median(counts)), 1)
        valid = np.all(counts >= runs_ok, axis=0)
        packed = np.sort(rs.scores, axis=2)[:, valid, :runs_ok]  # NaNs sort last
        bal = ResultSet(
            rs.models, tuple(t for t, v in zip(rs.tasks, valid, strict=True) if v), packed
        )
    else:
        bal = rs
    out: dict[str, Any] = {
        "n_models": rs.n_models,
        "n_tasks": rs.n_tasks,
        "n_runs": rs.n_runs,
        "balanced": balanced,
        "balanced_tasks": bal.n_tasks,
        "balanced_runs": bal.n_runs,
        "level": level,
    }
    ks = _ks(bal.n_runs)
    models = []
    for name in rs.models:
        m = rs.model(name)
        iv = bootstrap_ci(m, batch_success, level=level, n_boot=n_boot, seed=seed)
        bm = bal.model(name)
        ph, pa = {}, {}
        for k in ks:
            a = bootstrap_ci(bm, batch_pass_hat_k(k), level=level, n_boot=n_boot, seed=seed)
            b = bootstrap_ci(bm, batch_pass_at_k(k), level=level, n_boot=n_boot, seed=seed)
            ph[k] = (a.estimate, a.lo, a.hi)
            pa[k] = (b.estimate, b.lo, b.hi)
        icc = decompose_single(bm).icc if bal.n_runs >= 2 and bal.n_tasks >= 2 else None
        models.append(ModelSummary(name, iv.estimate, iv.lo, iv.hi, ph, pa, icc))
    out["models"] = [asdict(m) for m in models]
    if bal.n_models >= 2 and bal.n_tasks >= 2:
        ru = rank_uncertainty(bal, n_boot=n_boot, level=level, seed=seed)
        out["ranking"] = {
            "table": ru.table(),
            "rank_probs": ru.rank_probs.tolist(),
            "prob_better": ru.prob_better.tolist(),
            "models": list(ru.models),
        }
        pairs = pairwise(bal, n_boot=min(n_boot, 2000), seed=seed)
        out["pairwise"] = [asdict(p) for p in pairs]
        out["fraction_indistinguishable"] = (
            float(np.mean([p.p_adjusted >= 0.05 for p in pairs])) if pairs else None
        )
        rho, rho5 = split_half_rank_correlation(bal, seed=seed)
        out["split_half_tau"] = {"mean": rho, "p05": rho5}
        if bal.n_runs >= 2:
            comps = decompose_array(bal.scores)
            out["variance"] = {
                "components": {
                    "model": comps.model,
                    "task": comps.task,
                    "model_task": comps.model_task,
                    "run": comps.run,
                },
                "shares": comps.shares(),
                "g_coefficient": comps.g_coefficient(),
            }
            dv = DiffVariance.from_components(comps)
            out["power"] = {
                "diff_variance": {"task": dv.task, "run": dv.run},
                "mde_current": mde(bal.n_tasks, bal.n_runs, dv),
                "target_delta": mde_target,
                "plan": [asdict(r) for r in plan(mde_target, dv, cost_task=0.0)],
                "tasks_needed_at_current_runs": required_tasks(mde_target, bal.n_runs, dv),
                "optimal_runs_if_task_costs_10_runs": optimal_runs(dv, cost_task=10, cost_run=1),
            }
    return out
