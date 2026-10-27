"""Generate every figure, table and macro used by the paper from results/*.json.

No number in latex/ is typed by hand: run ``python scripts/make_paper_assets.py`` and recompile.
Sections that depend on the local-model runs are produced when results/real_main.json exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path("results")
OUT = Path("latex")
FIG = OUT / "figures"
GEN = OUT / "generated"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6b6b70"
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 150,
    }
)
MACROS: dict[str, str] = {}


def load(name: str) -> dict:  # type: ignore[type-arg]
    return json.loads((R / name).read_text())


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}\\%"


def macro(name: str, value: str) -> None:
    MACROS[name] = value


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / name, bbox_inches="tight")
    fig.savefig(Path("docs/assets") / name.replace(".pdf", ".svg"), bbox_inches="tight")
    plt.close(fig)


def sim_coverage() -> None:
    d = load("sim_coverage.json")
    cells = d["mean_score"]
    methods = {
        "wilson_pooled": ("Pooled Wilson (naive)", ORANGE, "-", "o"),
        "bootstrap_tasks": ("Task bootstrap", BLUE, "-", "s"),
        "beta_binomial_profile": ("Hierarchical beta-binomial", AQUA, "-", "^"),
        "bootstrap_two_stage": ("Two-stage bootstrap", GREY, "--", "x"),
        "bootstrap_runs_only": ("Runs-only bootstrap", GREY, ":", "d"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, t in zip(axes, (50, 300), strict=True):
        runs = sorted({c["runs"] for c in cells if c["tasks"] == t})
        for key, (lab, col, ls, mk) in methods.items():
            ys = [
                next(c for c in cells if c["tasks"] == t and c["runs"] == r)["coverage"][key]
                for r in runs
            ]
            ax.plot(runs, ys, color=col, ls=ls, marker=mk, ms=4, label=lab)
        ax.axhline(0.95, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(runs, [str(r) for r in runs])
        ax.set_xlabel("runs per task $R$")
        ax.set_title(f"$T={t}$ tasks")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("coverage of 95% interval")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=7,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.17),
    )
    save(fig, "coverage.pdf")
    pick = {(c["tasks"], c["runs"]): c for c in cells}
    for key, nm in (
        ("wilson_pooled", "Wilson"),
        ("clopper_pearson_pooled", "CP"),
        ("bootstrap_tasks", "Boot"),
        ("bootstrap_tasks_bca", "BCa"),
        ("beta_binomial_profile", "BB"),
        ("bootstrap_two_stage", "Two"),
        ("bootstrap_runs_only", "Runs"),
    ):
        macro(f"cov{nm}TenRuns", pct(pick[(100, 10)]["coverage"][key]))
    macro("covWilsonThreeRuns", pct(pick[(100, 3)]["coverage"]["wilson_pooled"]))
    macro(
        "covBootRange",
        f"{pct(min(c['coverage']['bootstrap_tasks'] for c in cells if c['runs'] > 1))}--{pct(max(c['coverage']['bootstrap_tasks'] for c in cells if c['runs'] > 1))}",
    )
    macro("simReps", str(cells[0]["reps"]))
    pk = d["pass_hat_k"]
    macro("covPassKMin", pct(min(c["coverage"] for c in pk)))
    macro("covPassKMax", pct(max(c["coverage"] for c in pk)))
    # tabular version
    rows = []
    for t in (20, 50, 100, 300):
        for r in (1, 3, 10):
            c = pick[(t, r)]["coverage"]
            rows.append(
                f"{t} & {r} & {100 * c['wilson_pooled']:.1f} & {100 * c['clopper_pearson_pooled']:.1f} & "
                f"{100 * c['bootstrap_tasks']:.1f} & {100 * c['bootstrap_tasks_bca']:.1f} & "
                f"{100 * c['bootstrap_two_stage']:.1f} & {100 * c['beta_binomial_profile']:.1f} \\\\"
            )
    head = (
        "\\begin{tabular}{rrrrrrrr}\n\\toprule\n"
        "$T$ & $R$ & Wilson & CP & Task boot. & BCa & Two-stage & Beta-bin. \\\\\n\\midrule\n"
    )
    (GEN / "tab_coverage.tex").write_text(
        head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )


def sim_tests() -> None:
    d = load("sim_tests.json")["cells"]
    sel = [c for c in d if c["delta_theta"] == 0.0 and c["tasks"] == 100]
    procs = [
        ("pooled_ztest", "Pooled $z$-test", ORANGE),
        ("pooled_mcnemar", "Pooled McNemar (runs)", "#b5481f"),
        ("task_permutation", "Task sign-flip", BLUE),
        ("task_bootstrap", "Task bootstrap", AQUA),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, sd in zip(axes, (0.5, 1.0), strict=True):
        runs = [1, 5, 10]
        w = 0.2
        for i, (k, lab, col) in enumerate(procs):
            ys = [
                next(c for c in sel if c["runs"] == r and c["interaction_sd"] == sd)["reject_rate"][
                    k
                ]
                for r in runs
            ]
            ax.bar(np.arange(3) + (i - 1.5) * w, ys, w, color=col, label=lab)
        ax.axhline(0.05, color="k", lw=0.8)
        ax.set_xticks(range(3), [str(r) for r in runs])
        ax.set_xlabel("runs per task $R$")
        ax.set_title(f"agent$\\times$task interaction sd = {sd}")
    axes[0].set_ylabel("type-I error at $\\alpha=0.05$")
    axes[1].legend(fontsize=7, frameon=False, loc="upper left")
    save(fig, "typeI.pdf")
    top = next(c for c in sel if c["runs"] == 10 and c["interaction_sd"] == 1.0)["reject_rate"]
    macro("typeIPoolZ", pct(top["pooled_ztest"]))
    macro("typeIPoolMc", pct(top["pooled_mcnemar"]))
    macro("typeIPerm", pct(top["task_permutation"]))
    macro("typeIBoot", pct(top["task_bootstrap"]))
    lo = next(c for c in sel if c["runs"] == 10 and c["interaction_sd"] == 0.5)["reject_rate"]
    macro("typeIPoolMcLow", pct(lo["pooled_mcnemar"]))
    allp = [c["reject_rate"]["task_permutation"] for c in d if c["delta_theta"] == 0.0]
    macro("typeIPermMax", pct(max(allp)))
    macro("typeIPermMin", pct(min(allp)))


def sim_design() -> None:
    d = load("sim_design.json")
    rows = [r for r in d["rows"] if r["budget"] == 1200]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    runs = [r["runs"] for r in rows]
    axes[0].plot(runs, [r["analytic_se"] for r in rows], color=BLUE, label="analytic")
    axes[0].plot(runs, [r["empirical_se"] for r in rows], "o", color=ORANGE, label="simulated")
    axes[0].set_xscale("log")
    axes[0].set_xticks(runs, [str(r) for r in runs])
    axes[0].set_xlabel("runs per task $R$ ($T=1200/R$)")
    axes[0].set_ylabel("SE of score difference")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].plot(runs, [r["analytic_power"] for r in rows], color=BLUE, label="analytic")
    axes[1].plot(runs, [r["power"] for r in rows], "o", color=ORANGE, label="simulated")
    axes[1].set_xscale("log")
    axes[1].set_xticks(runs, [str(r) for r in runs])
    axes[1].set_xlabel("runs per task $R$ ($T=1200/R$)")
    axes[1].set_ylabel("power")
    save(fig, "design.pdf")
    first, last = rows[0], rows[-1]
    macro("designBudget", "1200")
    macro("designPowerOne", pct(first["power"]))
    macro("designPowerTen", pct(next(r for r in rows if r["runs"] == 10)["power"]))
    macro("designPowerTwenty", pct(last["power"]))
    macro("designSEratio", f"{last['empirical_se'] / first['empirical_se']:.2f}")
    macro(
        "designMaxSEgap", pct(max(abs(r["empirical_se"] / r["analytic_se"] - 1) for r in d["rows"]))
    )


def sim_sequential() -> None:
    d = load("sim_sequential.json")["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    x = np.arange(len(d))
    w = 0.27
    for i, (k, lab, col) in enumerate(
        (
            ("seq", "Confidence sequence", BLUE),
            ("fixed", "Fixed-$n$ test", AQUA),
            ("peek", "Naive peeking", ORANGE),
        )
    ):
        axes[0].bar(x + (i - 1) * w, [r["decision_rate"][k] for r in d], w, color=col, label=lab)
    axes[0].set_xticks(x, [f"{r['true_diff'] * 100:.1f}" for r in d])
    axes[0].set_xlabel("true score difference (points)")
    axes[0].set_ylabel("rate of 'significant'")
    axes[0].axhline(0.05, color="k", lw=0.8)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    med = [r["median_tasks_used_when_stopped"] or np.nan for r in d]
    axes[1].bar(x, med, 0.5, color=BLUE)
    axes[1].axhline(d[0]["task_budget"], color="k", lw=0.8)
    axes[1].set_xticks(x, [f"{r['true_diff'] * 100:.1f}" for r in d])
    axes[1].set_xlabel("true score difference (points)")
    axes[1].set_ylabel("median tasks used when stopped")
    save(fig, "sequential.pdf")
    macro("seqNullPeek", pct(d[0]["decision_rate"]["peek"]))
    macro("seqNullSeq", pct(d[0]["decision_rate"]["seq"]))
    macro("seqNullFixed", pct(d[0]["decision_rate"]["fixed"]))
    macro("seqPowerSeqMid", pct(d[2]["decision_rate"]["seq"]))
    macro("seqPowerFixedMid", pct(d[2]["decision_rate"]["fixed"]))
    macro("seqMidDiff", f"{d[2]['true_diff'] * 100:.1f}")
    macro("seqLargeMedianTasks", f"{d[3]['median_tasks_used_when_stopped']:.0f}")
    macro("seqBudget", str(d[0]["task_budget"]))


def sim_ranking() -> None:
    d = load("sim_ranking.json")["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for row, col, ls in zip(d, (ORANGE, BLUE, AQUA), ("-", "--", ":"), strict=True):
        cal = row["calibration"]
        axes[0].plot(
            [c["mean_pred"] for c in cal],
            [c["observed"] for c in cal],
            marker="o",
            ms=3,
            color=col,
            ls=ls,
            label=f"$T={row['tasks']}$",
        )
    axes[0].plot([0, 1], [0, 1], color="k", lw=0.8)
    axes[0].set_xlabel("bootstrap $P(\\mathrm{best})$")
    axes[0].set_ylabel("observed frequency of being best")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].bar(
        [str(r["tasks"]) for r in d], [r["mean_coverage"] for r in d], color=BLUE, width=0.5
    )
    axes[1].axhline(0.95, color="k", lw=0.8)
    axes[1].set_ylim(0.9, 1.0)
    axes[1].set_xlabel("tasks $T$")
    axes[1].set_ylabel("coverage of 95% rank interval")
    save(fig, "ranking_calibration.pdf")
    macro("rankCovMin", pct(min(r["mean_coverage"] for r in d)))
    macro("rankCovMax", pct(max(r["mean_coverage"] for r in d)))


def swebench() -> None:
    d = load("reanalysis_swebench.json")
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.75}
    )
    g = d["by_score_gap"]
    labs = [f"{100 * x['gap_lo']:.0f}--{100 * x['gap_hi']:.0f}" for x in g]
    x = np.arange(len(g))
    axes[0].bar(
        x - 0.2, [v["frac_distinguishable_raw"] for v in g], 0.4, color=ORANGE, label="unadjusted"
    )
    axes[0].bar(
        x + 0.2, [v["frac_distinguishable_holm"] for v in g], 0.4, color=BLUE, label="Holm-adjusted"
    )
    axes[0].set_xticks(x, labs)
    axes[0].set_xlabel("score gap between two agents (points)")
    axes[0].set_ylabel("fraction of pairs distinguishable")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    top = d["top10"]
    y = np.arange(len(top))[::-1]
    axes[1].hlines(y, [t["rank_lo"] for t in top], [t["rank_hi"] for t in top], color=BLUE, lw=3)
    axes[1].plot([i + 1 for i in range(len(top))], y, "o", color=ORANGE, ms=4)
    axes[1].set_yticks(y, [f"#{i + 1}  {100 * t['score']:.1f}%" for i, t in enumerate(top)])
    axes[1].set_xlabel("95% bootstrap interval of the rank")
    save(fig, "swebench.pdf")
    macro("sweNAgents", str(d["n_agents"]))
    macro("sweNTasks", str(d["n_tasks"]))
    macro("sweNPairs", f"{d['n_pairs']:,}".replace(",", "\\,"))
    macro("sweFracHolm", pct(d["frac_distinguishable_holm"]))
    macro("sweFracRaw", pct(d["frac_distinguishable_raw"]))
    macro("sweFracBH", pct(d["frac_distinguishable_bh"]))
    macro("sweIndistHolm", pct(1 - d["frac_distinguishable_holm"]))
    macro("sweAdjHolm", pct(d["adjacent_pairs"]["frac_distinguishable_holm_within_adjacent"]))
    macro("sweAdjRaw", pct(d["adjacent_pairs"]["frac_distinguishable_raw"]))
    macro("sweAdjN", str(d["adjacent_pairs"]["n"]))
    macro("sweMinGap", f"{100 * d['min_gap_distinguishable']:.1f}")
    macro("sweMedGap", f"{100 * d['median_gap_needed']:.1f}")
    macro("sweTopScore", pct(top[0]["score"]))
    macro("sweTopRankHi", f"{top[0]['rank_hi']:.0f}")
    macro("sweTopPBest", pct(top[0]["p_best"], 0))
    macro("sweGapSmallHolm", pct(g[1]["frac_distinguishable_holm"], 0))
    macro("sweGapMidHolm", pct(g[2]["frac_distinguishable_holm"], 0))


def taubench() -> None:
    d = load("reanalysis_taubench.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    for ax, dom in zip(axes, ("airline", "retail"), strict=True):
        a = d[dom]["analysis"]
        for m, col in zip(a["models"], (ORANGE, BLUE), strict=True):
            ks = sorted(int(k) for k in m["pass_hat_k"])
            ax.plot(
                ks,
                [m["pass_hat_k"][str(k)][0] for k in ks],
                "-o",
                color=col,
                ms=4,
                label=f"{m['model']} pass^k",
            )
            ax.plot(
                ks,
                [m["pass_at_k"][str(k)][0] for k in ks],
                "--s",
                color=col,
                ms=3,
                label=f"{m['model']} pass@k",
            )
        ax.set_title(f"{dom} ($T={d[dom]['n_tasks']}$)")
        ax.set_xlabel("$k$")
        ax.set_xticks(range(1, d[dom]["n_runs"] + 1))
    axes[0].set_ylabel("probability")
    axes[1].legend(fontsize=6, frameon=False, loc="center right")
    save(fig, "taubench.pdf")
    for dom in ("airline", "retail"):
        c = d[dom]["task_level_comparison"]
        p = d[dom]["pooled_shortcut"]
        macro(f"tau{dom.capitalize()}Diff", f"{100 * c['diff']:+.1f}")
        macro(f"tau{dom.capitalize()}CI", f"[{100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f}]")
        macro(f"tau{dom.capitalize()}PPerm", f"{c['p_permutation']:.3f}")
        macro(f"tau{dom.capitalize()}PPool", f"{p['pooled_ztest_p']:.3f}")
        macro(f"tau{dom.capitalize()}PMc", f"{p['pooled_run_mcnemar_p']:.3f}")
        macro(f"tau{dom.capitalize()}Tasks", str(d[dom]["n_tasks"]))
        macro(
            f"tau{dom.capitalize()}Need",
            str(d[dom]["tasks_needed_for_80pct_power_at_observed_gap"]),
        )
        v = d[dom]["analysis"]["variance"]["shares"]
        macro(f"tau{dom.capitalize()}RunShare", pct(v["run"], 0))
        macro(f"tau{dom.capitalize()}TaskShare", pct(v["task"], 0))
        macro(f"tau{dom.capitalize()}InterShare", pct(v["model_task"], 0))


def static_assets() -> None:
    from agentaudit.agents.base import SYSTEM_PROMPT
    from agentaudit.agents.text_protocol import tools_prompt
    from agentaudit.bench.base import Param, ToolSpec
    from agentaudit.bench.suite import core_suite, generate_suite

    tasks = generate_suite(60, 0)
    macro("Ntasks", str(len(tasks)))
    macro("Ncore", str(len(core_suite(tasks))))
    import textwrap

    (GEN / "system_prompt.txt").write_text(textwrap.fill(SYSTEM_PROMPT, 88) + "\n")
    example = [ToolSpec("TOOL_NAME", "DESCRIPTION", {"ARG": Param("string", "DESCRIPTION")})]
    wrapped = "\n".join(
        textwrap.fill(line, 88, subsequent_indent="    ") if line else ""
        for line in tools_prompt(example).strip().splitlines()
    )
    (GEN / "tools_prompt.txt").write_text(wrapped + "\n")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    static_assets()
    sim_coverage()
    sim_tests()
    sim_design()
    sim_sequential()
    sim_ranking()
    swebench()
    taubench()
    from extra_assets import sim_extra, swebench_robustness

    swebench_robustness(macro, GEN)
    sim_extra(macro, save)
    if (R / "real_main.json").exists():
        from make_real_assets import real_assets

        real_assets(MACROS, FIG, GEN)
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())]
    (GEN / "macros.tex").write_text(
        "% generated by scripts/make_paper_assets.py\n" + "\n".join(lines) + "\n"
    )
    print(f"{len(MACROS)} macros, {len(list(FIG.glob('*.pdf')))} figures")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    main()
