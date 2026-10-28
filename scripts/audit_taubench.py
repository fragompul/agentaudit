# ruff: noqa: E501, C408
"""Final audit B: sensitivity of the tau-bench retail/airline inference to the analysis choice.

Rebuilds the paired data from the raw trajectories in data/raw/taubench (gpt-4o vs
claude-3.5-sonnet, 4 trials/task) and applies every procedure to the same data. Writes
results/audit_taubench.json. Primary analysis (declared before looking at the table): estimand =
difference of benchmark scores (mean over tasks of per-task mean reward); unit = task; test =
studentised sign-flip; interval = task bootstrap; Holm across the two domains.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from agentaudit.compare import (
    holm,
    mcnemar_exact,
    paired_bootstrap,
    sign_flip_permutation,
    studentized_sign_flip,
)
from agentaudit.ingest import taubench_resultset

D = Path("data/raw/taubench")
NPERM = 100_000


def label_permutation_p(a: np.ndarray, b: np.ndarray, n_perm: int = NPERM, seed: int = 0) -> float:
    """Within-task permutation of the agent label over all 2R runs: sharp null that the two agents
    have the same outcome distribution on every task (stronger than equal benchmark means)."""
    t, r = a.shape
    both = np.concatenate([a, b], axis=1)
    obs = abs(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm // 1000):
        keys = rng.random((1000, t, 2 * r))
        idx = np.argsort(keys, axis=2)
        perm = np.take_along_axis(np.broadcast_to(both, (1000, t, 2 * r)), idx, axis=2)
        diff = np.abs(perm[:, :, :r].mean(axis=(1, 2)) - perm[:, :, r:].mean(axis=(1, 2)))
        hits += int(np.sum(diff >= obs - 1e-12))
    n = (n_perm // 1000) * 1000
    return (1 + hits) / (n + 1)


def domain(name: str) -> dict:
    rs = taubench_resultset(
        {"gpt-4o": D / f"gpt-4o-{name}.json", "sonnet": D / f"sonnet-35-new-{name}.json"}
    )
    a, b = rs.model("sonnet"), rs.model("gpt-4o")  # (tasks, runs); diff = sonnet - gpt-4o
    t, r = a.shape
    d = a.mean(axis=1) - b.mean(axis=1)
    rows = {}
    lo, hi, pb = paired_bootstrap(d, n_boot=20_000, seed=1)
    rows["task_bootstrap_percentile"] = dict(est=d.mean(), lo=lo, hi=hi, p=pb)
    rows["sign_flip_sum"] = dict(p=sign_flip_permutation(d, n_perm=NPERM, seed=1))
    rows["studentised_sign_flip"] = dict(p=studentized_sign_flip(d, n_perm=NPERM, seed=1))
    tt = stats.ttest_1samp(d, 0.0)
    rows["paired_t"] = dict(p=float(tt.pvalue))
    nz = d[d != 0]
    rows["wilcoxon_signed_rank"] = dict(p=float(stats.wilcoxon(nz).pvalue), n_nonzero=int(nz.size))
    rows["label_permutation_within_task"] = dict(p=label_permutation_p(a, b, seed=1))
    # majority outcome (strictly > 1/2; 2/4 counts as a failure) and the tie-excluding variant
    ma, mb = a.mean(axis=1), b.mean(axis=1)
    sa, sb = ma > 0.5, mb > 0.5
    rows["mcnemar_majority_tie_as_fail"] = dict(
        p=mcnemar_exact(int((sa & ~sb).sum()), int((sb & ~sa).sum())),
        discordant=[int((sa & ~sb).sum()), int((sb & ~sa).sum())],
        est=float(sa.mean() - sb.mean()),
    )
    keep = (ma != 0.5) & (mb != 0.5)
    rows["mcnemar_majority_ties_dropped"] = dict(
        p=mcnemar_exact(int((sa & ~sb & keep).sum()), int((sb & ~sa & keep).sum())),
        n_ties_dropped=int((~keep).sum()),
    )
    # "all trials succeed" (pass^4) and "any trial succeeds" (pass@4) are other estimands
    for lab, fa, fb in (("pass_hat4", ma == 1, mb == 1), ("pass_at4", ma > 0, mb > 0)):
        rows[f"mcnemar_{lab}"] = dict(
            p=mcnemar_exact(int((fa & ~fb).sum()), int((fb & ~fa).sum())),
            est=float(fa.mean() - fb.mean()),
        )
    # pooled shortcuts (invalid: treat 4 runs of a task as independent)
    n = a.size
    pa, pb_ = a.mean(), b.mean()
    pp = (pa + pb_) / 2
    rows["pooled_z_runs_independent"] = dict(
        p=float(2 * stats.norm.sf(abs(pa - pb_) / np.sqrt(2 * pp * (1 - pp) / n)))
    )
    # GEE, cluster = task; identity link estimates the same risk difference with a sandwich SE
    long = pd.DataFrame(
        {
            "y": np.concatenate([a.ravel(), b.ravel()]),
            "sonnet": np.r_[np.ones(a.size), np.zeros(b.size)],
            "task": np.r_[np.repeat(np.arange(t), r), np.repeat(np.arange(t), r)],
        }
    )
    ex = sm.cov_struct.Exchangeable()
    g1 = smf.gee("y ~ sonnet", "task", long, family=sm.families.Gaussian(), cov_struct=ex).fit()
    rows["gee_identity_cluster_task"] = dict(
        est=float(g1.params["sonnet"]),
        lo=float(g1.conf_int().loc["sonnet", 0]),
        hi=float(g1.conf_int().loc["sonnet", 1]),
        p=float(g1.pvalues["sonnet"]),
    )
    g2 = smf.gee("y ~ sonnet", "task", long, family=sm.families.Binomial(), cov_struct=ex).fit()
    rows["gee_logit_cluster_task_marginal_OR"] = dict(
        est_odds_ratio=float(np.exp(g2.params["sonnet"])), p=float(g2.pvalues["sonnet"])
    )
    # conditional logistic-type check: task fixed effects (GLM with task dummies, cluster-robust)
    g3 = smf.glm("y ~ sonnet + C(task)", long, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": long["task"]}
    )
    rows["glm_logit_task_fixed_effects_cluster_robust"] = dict(
        est_odds_ratio=float(np.exp(g3.params["sonnet"])), p=float(g3.pvalues["sonnet"])
    )
    skew = float(stats.skew(d))
    return {
        "n_tasks": t,
        "runs": r,
        "est_diff": float(d.mean()),
        "n_nonzero_task_diffs": int((d != 0).sum()),
        "task_diff_skewness": skew,
        "rows": {
            k: {kk: (vv if not isinstance(vv, float) else float(vv)) for kk, vv in v.items()}
            for k, v in rows.items()
        },
    }


def main() -> None:
    out = {dom: domain(dom) for dom in ("retail", "airline")}
    prim = {dom: out[dom]["rows"]["studentised_sign_flip"]["p"] for dom in out}
    h = holm([prim["retail"], prim["airline"]])
    out["primary_holm_studentised_sign_flip"] = {"retail": float(h[0]), "airline": float(h[1])}
    for meth in ("sign_flip_sum", "label_permutation_within_task", "mcnemar_majority_tie_as_fail"):
        hh = holm([out["retail"]["rows"][meth]["p"], out["airline"]["rows"][meth]["p"]])
        out[f"holm_{meth}"] = {"retail": float(hh[0]), "airline": float(hh[1])}
    Path("results/audit_taubench.json").write_text(json.dumps(out, indent=1, default=float))
    for dom in ("retail", "airline"):
        print(
            dom, out[dom]["n_tasks"], out[dom]["est_diff"], "skew", out[dom]["task_diff_skewness"]
        )
        for k, v in out[dom]["rows"].items():
            print("  ", k, v)
    print({k: v for k, v in out.items() if k.startswith("holm") or k.startswith("primary")})


if __name__ == "__main__":
    main()
