"""Figures, tables and macros for the extra experiments (E1-E6) and the SWE-bench robustness analysis.

Imported by ``make_paper_assets.py``; kept apart so that each module stays readable.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

R = Path("results")
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6b6b70"


def _load(name: str) -> Any:
    return json.loads((R / name).read_text())


def _pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}\\%"


def sim_extra(macro: Callable[[str, str], None], save: Callable[[plt.Figure, str], None]) -> None:
    """Experiments E1, E2, E3, E5, E6 (pre-specified in docs/experiments.md)."""
    # E1: weak null with skewed paired differences
    a = _load("sim_asym.json")["cells"]
    meths = ("sign_flip", "studentized_sign_flip", "paired_bootstrap", "paired_t")
    macro("asymReps", str(a[0]["reps"]))
    macro("asymMaxAny", _pct(max(max(c["reject_rate"].values()) for c in a)))
    for k, nm in zip(meths, ("SF", "SSF", "Boot", "T"), strict=True):
        macro(f"asymMax{nm}", _pct(max(c["reject_rate"][k] for c in a)))
        macro(f"asymMean{nm}", _pct(float(np.mean([c["reject_rate"][k] for c in a]))))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, q in zip(axes, (0.5, 0.1, 0.03), strict=True):
        for k, lab, col in zip(
            meths,
            ("Sign-flip", "Studentised sign-flip", "Paired bootstrap", "Paired $t$"),
            (BLUE, AQUA, ORANGE, GREY),
            strict=True,
        ):
            pts = [(c["tasks"], c["reject_rate"][k]) for c in a if c["q_big_win"] == q and c["runs"] == 5]
            ax.plot(*zip(*pts, strict=True), "-o", ms=3, color=col, label=lab)
        ax.axhline(0.05, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks([20, 50, 100, 300], ["20", "50", "100", "300"])
        sk = next(c for c in a if c["q_big_win"] == q)["skewness_of_delta"]
        ax.set_title(f"skewness {sk:.1f}")
        ax.set_xlabel("tasks $T$")
    axes[0].set_ylabel("type-I error ($R=5$)")
    axes[0].legend(fontsize=6, frameon=False, loc="lower right")
    save(fig, "asymmetric_null.pdf")

    # E2: allocation
    al = _load("sim_alloc.json")["cells"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, st in zip(axes, (0.5, 1.5, 2.5), strict=True):
        for si, col in zip((0.2, 0.6, 1.0), (AQUA, BLUE, ORANGE), strict=True):
            c = next(x for x in al if x["sigma_task"] == st and x["sigma_interaction"] == si)
            fb = c["fixed_budget"]
            base = fb[0]["empirical_se"]
            ax.plot(
                [r["runs"] for r in fb],
                [r["empirical_se"] / base for r in fb],
                "-o",
                ms=3,
                color=col,
                label=f"interaction sd {si}",
            )
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
        ax.set_title(f"task sd {st}")
        ax.set_xlabel("runs per task $R$")
    axes[0].set_ylabel("SE relative to $R=1$")
    axes[0].legend(fontsize=6, frameon=False)
    save(fig, "allocation.pdf")

    def ratio(c: dict[str, Any]) -> float:
        return float(c["fixed_budget"][3]["empirical_se"] / c["fixed_budget"][0]["empirical_se"])

    low = [ratio(c) for c in al if c["sigma_interaction"] == 0.2]
    high = [ratio(c) for c in al if c["sigma_interaction"] == 1.0]
    macro("allocRatioLow", f"{min(low):.2f}--{max(low):.2f}")
    macro("allocRatioHigh", f"{min(high):.2f}--{max(high):.2f}")
    ten = lambda c: c["optimal_runs_by_cost_ratio"]["10.0"]  # noqa: E731
    macro("allocRstarTenHigh", f"{ten(next(c for c in al if c['sigma_interaction'] == 1.0)):.0f}")
    macro("allocRstarTenLow", f"{ten(next(c for c in al if c['sigma_interaction'] == 0.2)):.0f}")
    macro("allocReps", str(al[0]["reps"]))
    gap = max(abs(r["empirical_se"] / r["analytic_se"] - 1) for c in al for r in c["fixed_budget"])
    macro("allocMaxSEgap", _pct(gap))

    # E3: robustness of coverage
    cv = _load("sim_covrob.json")["cells"]
    macro("covrobReps", str(cv[0]["reps"]))
    hi = [c for c in cv if c["sigma_task"] == 1.5 and c["runs"] == 10]
    macro("covrobWilsonMin", _pct(min(c["coverage"]["wilson_pooled"] for c in hi), 0))
    macro("covrobWilsonMax", _pct(max(c["coverage"]["wilson_pooled"] for c in hi), 0))
    big = [c for c in cv if c["tasks"] >= 50 and c["runs"] > 1]
    macro("covrobBootMin", _pct(min(c["coverage"]["bootstrap_tasks"] for c in big)))
    macro("covrobBootMax", _pct(max(c["coverage"]["bootstrap_tasks"] for c in big)))
    macro("covrobBBMin", _pct(min(c["coverage"]["beta_binomial_profile"] for c in big)))
    macro("covrobBBMax", _pct(max(c["coverage"]["beta_binomial_profile"] for c in big)))
    small = [
        c
        for c in cv
        if c["tasks"] == 20 and c["runs"] == 1 and c["sigma_task"] == 0.3 and c["theta"] != 0.3
    ]
    macro("covrobBootSmallMin", _pct(min(c["coverage"]["bootstrap_tasks"] for c in small), 0))
    macro("covrobBootSmallMax", _pct(max(c["coverage"]["bootstrap_tasks"] for c in small), 0))
    macro("covrobBCaSmallMin", _pct(min(c["coverage"]["bootstrap_tasks_bca"] for c in small), 0))
    macro("covrobRunsOnlyOne", _pct(max(c["coverage"]["bootstrap_runs_only"] for c in cv if c["runs"] == 1), 0))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, st in zip(axes, (0.3, 1.5), strict=True):
        for key, lab, col in (
            ("wilson_pooled", "Pooled Wilson", ORANGE),
            ("bootstrap_tasks", "Task bootstrap", BLUE),
            ("bootstrap_tasks_bca", "Task BCa", GREY),
            ("beta_binomial_profile", "Beta-binomial", AQUA),
        ):
            for th, mk in zip((-2.5, 0.3, 2.5), ("v", "o", "^"), strict=True):
                pts = [
                    (c["tasks"], c["coverage"][key])
                    for c in cv
                    if c["sigma_task"] == st and c["runs"] == 10 and c["theta"] == th
                ]
                ax.plot(
                    *zip(*pts, strict=True), ls="-", marker=mk, ms=3, color=col, label=lab if th == 0.3 else None
                )
        ax.axhline(0.95, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks([20, 50, 100], ["20", "50", "100"])
        ax.set_xlabel("tasks $T$ ($R=10$)")
        ax.set_title(f"task sd {st}; markers: low / mid / high prevalence", fontsize=7)
    axes[0].set_ylabel("coverage of 95% interval")
    axes[0].legend(fontsize=6, frameon=False, loc="lower right")
    save(fig, "coverage_robustness.pdf")

    # E5: sequential margins
    sq = _load("sim_seqm.json")["cells"]
    macro("seqmReps", str(sq[0]["reps"]))
    outside = [c for c in sq if c["true_diff"] > c["margin"] > 0]
    macro("seqmFalseEquivMax", _pct(max(c["rates"]["equivalent"] for c in outside)))
    tie = next(c for c in sq if c["delta_theta"] == 0.0 and c["margin"] == 0.1 and c["min_tasks"] == 10)
    macro("seqmEquivTrueTie", _pct(tie["rates"]["equivalent"], 0))
    macro("seqmEquivTrueTieTasks", f"{tie['median_tasks_when_decided']:.0f}")
    macro("seqmZeroMarginEquiv", _pct(max(c["rates"]["equivalent"] for c in sq if c["margin"] == 0)))
    t0 = next(c for c in sq if c["delta_theta"] == 0.0 and c["margin"] == 0.0 and c["min_tasks"] == 10)
    macro("seqmUndecidedTie", _pct(t0["rates"]["undecided"], 0))

    # E6: pairwise Holm
    rk = _load("sim_rankpw.json")["cells"]
    macro("rankpwReps", str(rk[0]["reps"]))
    macro("rankpwFWE", _pct(max(c["false_rejection_rate_equal_pair"] for c in rk)))

    def pair02(c: dict[str, Any]) -> dict[str, Any]:
        return next(p for p in c["pairs"] if (p["i"], p["j"]) == (0, 2))

    large = next(c for c in rk if c["tasks"] == 500 and c["runs"] == 3)
    macro("rankpwCloseTrue", f"{100 * pair02(large)['true_abs_diff']:.1f}")
    macro("rankpwCloseRej", _pct(pair02(large)["holm_reject_rate"], 0))
    macro("rankpwCloseRejSmall", _pct(pair02(next(c for c in rk if c["tasks"] == 50 and c["runs"] == 3))["holm_reject_rate"], 0))


def swebench_robustness(macro: Callable[[str, str], None], gen: Path) -> None:
    d = _load("reanalysis_swebench.json")
    ca = d["coverage_audit"]
    macro("sweNCovered", str(ca["coverage_verified"]))
    macro("sweNUnverif", str(ca["coverage_unverifiable_no_public_logs"]))
    macro("sweNIncomplete", str(ca["coverage_incomplete"]))
    macro("sweNSingle", str(ca["attempts_1"]))
    macro("sweNMulti", str(ca["attempts_multiple"]))
    macro("sweNAttUnknown", str(ca["attempts_unknown"]))
    rows = []
    tags = {"all": "All", "covered": "Cov", "single": "Single", "covered+single": "Both"}
    labels = {
        "all": "All entries",
        "covered": "Coverage verified",
        "single": "One attempt declared",
        "covered+single": "Both",
    }
    for key, tag in tags.items():
        s = d["subsets"][key]
        ad = s["adjacent_pairs"]
        adj = f"{ad['n_distinguishable_holm_all_pairs']}/{ad['n']}"
        raw = f"{ad['n_distinguishable_raw']}/{ad['n']}"
        rows.append(
            f"{labels[key]} & {s['n_agents']} & {100 * s['frac_distinguishable_holm']:.1f} & {adj} & {raw} \\\\"
        )
        macro(f"swe{tag}N", str(s["n_agents"]))
        macro(f"swe{tag}FracHolm", _pct(s["frac_distinguishable_holm"]))
        macro(f"swe{tag}AdjHolm", adj)
        macro(f"swe{tag}AdjRaw", raw)
    head = (
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Subset & Entries & Pairs separable (\\%) & Adjacent, Holm & Adjacent, unadjusted \\\\\n\\midrule\n"
    )
    (gen / "tab_swe_robust.tex").write_text(head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
