"""Figures, tables and macros for the local-model evaluation (called by make_paper_assets.py).

Reads results/real_main.json (analyze_real.py) and, when present, results/ablation_main.json
(ablate_real.py). Every number in the paper text about local models comes from here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

R = Path("results")
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6b6b70"
CYCLE = [BLUE, ORANGE, AQUA, "#8a5cd0", "#c9a227", GREY, "#d0467a"]


def _pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}\\%"


def _tex(s: str) -> str:
    return s.replace("_", "\\_")


def _save(fig: plt.Figure, fig_dir: Path, name: str) -> None:
    fig.savefig(fig_dir / name, bbox_inches="tight")
    fig.savefig(Path("docs/assets") / name.replace(".pdf", ".svg"), bbox_inches="tight")
    plt.close(fig)


def _main_figure(d: dict[str, Any], fig_dir: Path) -> None:
    a = d["analysis"]
    models = sorted(a["models"], key=lambda m: m["score"])
    fig, axes = plt.subplots(
        1, 3, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [1.25, 1, 0.8], "wspace": 0.55}
    )
    y = np.arange(len(models))
    axes[0].hlines(y, [m["lo"] for m in models], [m["hi"] for m in models], color=BLUE, lw=3)
    axes[0].plot([m["score"] for m in models], y, "o", color=ORANGE, ms=4)
    axes[0].set_yticks(y, [m["model"] for m in models], fontsize=7)
    axes[0].set_xlabel("success rate (95% task-bootstrap CI)")
    ks = sorted(int(k) for k in models[0]["pass_hat_k"])
    for i, m in enumerate(sorted(models, key=lambda m: -m["score"])):
        axes[1].plot(ks, [m["pass_hat_k"][str(k)][0] for k in ks], "-o", ms=3, color=CYCLE[i % 7])
    axes[1].set_xscale("log")
    axes[1].set_xticks(ks, [str(k) for k in ks])
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("pass$^k$")
    sh = a["variance"]["shares"]
    keys = [
        ("task", "task"),
        ("model_task", "agent×task"),
        ("run", "run noise"),
        ("model", "agent"),
    ]
    left = 0.0
    for i, (k, lab) in enumerate(keys):
        axes[2].barh([0], [sh[k]], left=left, color=CYCLE[i], height=0.5, label=lab)
        left += sh[k]
    axes[2].set_yticks([])
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel("share of variance")
    axes[2].legend(
        frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2
    )
    axes[2].grid(False)
    _save(fig, fig_dir, "real_main.pdf")


def _profile_figure(d: dict[str, Any], fig_dir: Path) -> None:
    prof, scores = d["profile"], {m["model"]: m["score"] for m in d["analysis"]["models"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"wspace": 0.35})
    for i, (name, p) in enumerate(sorted(prof.items())):
        c = CYCLE[i % 7]
        axes[0].errorbar(
            p["mean_tokens"] / 1000,
            scores[name],
            yerr=[[scores[name] - p["score_ci"][0]], [p["score_ci"][1] - scores[name]]],
            fmt="o",
            color=c,
            ms=4,
            capsize=2,
            label=name,
        )
        axes[1].plot(p["mean_latency_s"], scores[name], "o", color=c, ms=4)
    axes[0].set_xlabel("mean tokens per attempt (thousands)")
    axes[0].set_ylabel("success rate")
    axes[1].set_xlabel("mean wall-clock per attempt (s, 4 parallel)")
    axes[0].legend(frameon=False, fontsize=6)
    _save(fig, fig_dir, "real_profile.pdf")


def _ablation_figure(ab: dict[str, Any], fig_dir: Path) -> None:
    grid = ab["subdesign"]
    runs = sorted({g["runs"] for g in grid})
    sel = [r for r in (1, 3, runs[-1]) if r in runs]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw={"wspace": 0.45})
    for i, r in enumerate(sel):
        rows = sorted((g for g in grid if g["runs"] == r), key=lambda g: g["tasks"])
        t = [g["tasks"] for g in rows]
        axes[0].plot(t, [g["tau_mean"] for g in rows], "-o", ms=3, color=CYCLE[i], label=f"R={r}")
        axes[1].plot(t, [g["frac_pairs_distinguishable"] for g in rows], "-o", ms=3, color=CYCLE[i])
        ag = [g["sign_agreement_on_full_significant"] for g in rows]
        if all(v is not None for v in ag):
            axes[2].plot(t, ag, "-o", ms=3, color=CYCLE[i])
    axes[1].axhline(ab["full"]["frac_pairs_distinguishable"], color=GREY, ls="--", lw=1)
    for ax, lab in zip(
        axes,
        ["Kendall τ with full-data ranking", "fraction of pairs separated", "sign agreement"],
        strict=True,
    ):
        ax.set_xlabel("tasks sampled")
        ax.set_ylabel(lab, fontsize=7)
    axes[0].legend(frameon=False, fontsize=7)
    _save(fig, fig_dir, "real_ablation.pdf")


def _appendix_table(d: dict[str, Any], gen_dir: Path) -> None:
    a, fam, prof = d["analysis"], d["by_family"], d["profile"]
    fams = list(next(iter(fam.values())))
    lines = [
        "\\begin{tabular}{l" + "r" * (5 + len(fams)) + "}",
        "\\toprule",
        "Model & score & pass$^3$ & pass$^{10}$ & steps & step-limit & "
        + " & ".join(_tex(f) for f in fams)
        + " \\\\",
        "\\midrule",
    ]
    for m in sorted(a["models"], key=lambda m: -m["score"]):
        p = prof[m["model"]]
        cells = [
            _tex(m["model"]),
            f"{100 * m['score']:.1f}",
            f"{100 * m['pass_hat_k']['3'][0]:.1f}",
            f"{100 * m['pass_hat_k']['10'][0]:.1f}",
            f"{p['mean_steps']:.1f}",
            f"{100 * p['frac_step_limit']:.0f}\\%",
            *[f"{100 * fam[m['model']][f]:.0f}" for f in fams],
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (gen_dir / "tab_appendix_real.tex").write_text("\n".join(lines) + "\n")


def real_assets(macros: dict[str, str], fig_dir: Path, gen_dir: Path) -> None:
    d = json.loads((R / "real_main.json").read_text())
    a = d["analysis"]
    models = sorted(a["models"], key=lambda m: -m["score"])
    best, worst = models[0], models[-1]
    var = a["variance"]
    sh = var["shares"]
    pw = a["power"]
    n_att = d["n_trajectories"]
    _main_figure(d, fig_dir)
    _profile_figure(d, fig_dir)
    _appendix_table(d, gen_dir)
    floor = [m["model"] for m in models if m["score"] < 0.10]
    macros.update(
        {
            "realNModels": str(len(models)),
            "realNTasks": str(d["n_tasks"]),
            "realNRuns": str(d["n_runs"]),
            "realNAttempts": f"{n_att:,}".replace(",", "\\,"),
            "realBestModel": _tex(best["model"]),
            "realBestScore": _pct(best["score"]),
            "realWorstModel": _tex(worst["model"]),
            "realWorstScore": _pct(worst["score"]),
            "realModelShare": _pct(sh["model"], 0),
            "realTaskShare": _pct(sh["task"], 0),
            "realInterShare": _pct(sh["model_task"], 0),
            "realRunShare": _pct(sh["run"], 0),
            "realG": f"{var['g_coefficient']:.2f}",
            "realFracSep": _pct(1 - a["fraction_indistinguishable"], 0),
            "realMDE": f"{100 * pw['mde_current']:.1f}",
            "realTasksNeeded": str(pw["tasks_needed_at_current_runs"]),
            "realOptRuns": str(pw["optimal_runs_if_task_costs_10_runs"]),
            "realSplitTau": f"{a['split_half_tau']['mean']:.2f}",
            "realNFloor": str(len(floor)),
        }
    )
    best_p10 = max(models, key=lambda m: m["pass_hat_k"]["10"][0])
    macros["realBestPassTen"] = _pct(best_p10["pass_hat_k"]["10"][0])
    macros["realBestPassThree"] = _pct(best["pass_hat_k"]["3"][0])
    macros["realBestPassOne"] = _pct(best["pass_hat_k"]["1"][0])

    pairs = a["pairwise"]
    close = min(pairs, key=lambda q: abs(q["diff"]))
    ns = [q for q in pairs if q["p_adjusted"] >= 0.05]
    macros.update(
        {
            "realNPairs": str(len(pairs)),
            "realNIndist": str(len(ns)),
            "realClosestA": _tex(close["a"]),
            "realClosestB": _tex(close["b"]),
            "realClosestDiff": f"{100 * abs(close['diff']):.1f}",
            "realClosestP": f"{close['p_adjusted']:.2f}",
        }
    )
    fails = d["failures"]
    macros["realFloorNoAction"] = _pct(
        max(v.get("no_action", 0) / (d["n_tasks"] * d["n_runs"]) for v in fails.values()), 0
    )
    macros["realMaxStepLimit"] = _pct(max(p["frac_step_limit"] for p in d["profile"].values()), 0)
    macros["realMaxStepLimitModel"] = _tex(
        max(d["profile"].items(), key=lambda kv: kv[1]["frac_step_limit"])[0]
    )

    ab_path = R / "ablation_main.json"
    ab = json.loads(ab_path.read_text()) if ab_path.exists() else None
    if ab is not None:
        _ablation_figure(ab, fig_dir)
        grid = ab["subdesign"]
        t_max = max(g["tasks"] for g in grid)
        one = [g for g in grid if g["runs"] == 1 and g["tasks"] == t_max][0]
        ten = [
            g for g in grid if g["runs"] == max(x["runs"] for x in grid) and g["tasks"] == t_max
        ][0]
        small = [g for g in grid if g["tasks"] == min(x["tasks"] for x in grid) and g["runs"] == 1][
            0
        ]
        macros.update(
            {
                "ablTauOneRun": f"{one['tau_mean']:.2f}",
                "ablTauAllRuns": f"{ten['tau_mean']:.2f}",
                "ablTauFewTasks": f"{small['tau_mean']:.2f}",
                "ablFewTasks": str(small["tasks"]),
                "ablFracOneRun": _pct(one["frac_pairs_distinguishable"], 0),
                "ablFracAllRuns": _pct(ten["frac_pairs_distinguishable"], 0),
                "ablFracFewTasks": _pct(small["frac_pairs_distinguishable"], 0),
            }
        )
        seq = ab["sequential"]
        dec = [s for s in seq if s["full_data_significant_holm"]]
        sav = [s["median_tasks_when_decided"] for s in dec if s["median_tasks_when_decided"]]
        macros["seqRealNSig"] = str(len(dec))
        macros["seqRealNPairs"] = str(len(seq))
        macros["seqRealMedTasks"] = f"{np.median(sav):.0f}" if sav else "n/a"
        macros["seqRealDecided"] = (
            _pct(float(np.mean([1 - s["decisions"]["undecided"] for s in dec])), 0)
            if dec
            else "n/a"
        )

    floor_txt = (
        f" {len(floor)} of the {len(models)} models are near the floor (below 10\\%), where "
        "differences are trivially separable but uninformative about agent quality."
        if floor
        else ""
    )
    (gen_dir / "real_macros.tex").write_text(
        "% generated by scripts/make_real_assets.py\n"
        "\\renewcommand{\\REALINTRO}{an evaluation of "
        + f"{len(models)} local open-weight models on {d['n_tasks']} tasks with {d['n_runs']} runs each"
        + "}\n"
        "\\renewcommand{\\REALPROTOCOL}{We evaluate "
        + f"{len(models)} open-weight models served by Ollama on a single laptop (Apple M5, 16\\,GB) "
        + f"on the {d['n_tasks']}-task core subset of \\textsc{{AgentAudit-Bench}}, "
        + f"{d['n_runs']} independent runs per task ({n_att:,} attempts), temperature 0.7, "
        + "with the text tool protocol of Appendix~\\ref{app:prompts} so that every model is driven "
        + "identically."
        + "}\n"
        "\\renewcommand{\\REALABSTRACT}{On "
        + f"{len(models)} local models ({d['n_runs']} runs on {d['n_tasks']} tasks) run-to-run noise "
        + f"accounts for {_pct(sh['run'], 0)} of score variance and a {pw['target_delta'] * 100:.0f}-point "
        + f"difference would need about {pw['tasks_needed_at_current_runs']} tasks."
        + "}\n"
        "\\renewcommand{\\REALREFUTED}{On local models the ranking is not fragile: with gaps of tens of points all but "
        + f"{macros['realNIndist']} of {macros['realNPairs']} pairs are separable and run noise does not reorder them, "
        + "so the claim that leaderboard rankings are generally unreliable is supported only for close systems (SWE-bench) and not for "
        + "widely spaced ones. We did not test the fixed-budget tasks-versus-runs result on real data, the real-data sequential "
        + "analysis re-orders existing results rather than stopping a live run, and one model (q2) is a floor-effect control. "
        + "Single hardware, single prompt protocol, small models: conclusions do not transfer to frontier agents."
        + "}\n"
        "\\renewcommand{\\REALFINDINGS}{(6) On local models, run noise is "
        + f"{_pct(sh['run'], 0)} of variance, the best model's pass$^{{10}}$ is "
        + f"{_pct(best['pass_hat_k']['10'][0])} against a one-run score of "
        + f"{_pct(best['score'])}."
        + floor_txt
        + "}\n"
    )
