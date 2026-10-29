# ruff: noqa: E501, C408
"""Tables and macros for the final audit (results/audit_*.json). Imported by make_paper_assets.py."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

R = Path("results")


def _load(name: str) -> Any:
    return json.loads((R / name).read_text())


def _pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}\\%"


def _p(x: float) -> str:
    if x < 5e-4:
        m, e = f"{x:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{x:.3f}"


# (key in results/audit_taubench.json, label, estimand / null, nature, limitation)
TAU_METHODS = [
    (
        "task_bootstrap_percentile",
        "Task bootstrap (20k)",
        "mean-score difference; $H_0$: bootstrap re-centred",
        "simulation",
        "interval is percentile; needs enough tasks",
    ),
    (
        "studentised_sign_flip",
        "Studentised sign-flip (primary)",
        "mean-score difference; $E[d]=0$",
        "permutation, MC (100k)",
        "asymptotic for skewed $d$; one-sample theory not proved",
    ),
    (
        "sign_flip_sum",
        "Plain sign-flip",
        "symmetric $d_t$ about 0",
        "permutation, MC (100k)",
        "exact only under symmetry",
    ),
    (
        "paired_t",
        "Paired $t$ on task means",
        "mean-score difference; $E[d]=0$",
        "asymptotic",
        "normal approximation, 115/50 tasks",
    ),
    (
        "wilcoxon_signed_rank",
        "Wilcoxon signed-rank",
        "symmetric $d_t$; zeros dropped",
        "asymptotic",
        "drops tasks with $d_t=0$; different null",
    ),
    (
        "label_permutation_within_task",
        "Within-task label permutation",
        "identical outcome law per task",
        "permutation, MC (100k)",
        "sharp null, stronger than equal means",
    ),
    (
        "gee_identity_cluster_task",
        "GEE, identity link, task clusters",
        "mean-score difference",
        "asymptotic (sandwich)",
        "equals task-level mean test in large samples",
    ),
    (
        "gee_logit_cluster_task_marginal_OR",
        "GEE, logit link, task clusters",
        "marginal odds ratio",
        "asymptotic (sandwich)",
        "different scale (odds ratio)",
    ),
    (
        "glm_logit_task_fixed_effects_cluster_robust",
        "Logistic, task fixed effects",
        "task-conditional odds ratio",
        "asymptotic, cluster-robust",
        "different estimand; many nuisance parameters",
    ),
    (
        "mcnemar_majority_tie_as_fail",
        "McNemar, majority (2/4 = fail)",
        "difference of majority-success rates",
        "exact binomial",
        "arbitrary tie rule; run information lost",
    ),
    (
        "mcnemar_majority_ties_dropped",
        "McNemar, majority, ties dropped",
        "same, on non-tied tasks",
        "exact binomial",
        "changes the task population",
    ),
    (
        "mcnemar_pass_hat4",
        "McNemar, pass$^4$",
        "difference of all-4-succeed rates",
        "exact binomial",
        "reliability estimand",
    ),
    (
        "mcnemar_pass_at4",
        "McNemar, pass@4",
        "difference of any-of-4 rates",
        "exact binomial",
        "ceiling-type estimand",
    ),
    (
        "pooled_z_runs_independent",
        "Pooled $z$ (runs independent)",
        "two proportions, runs as units",
        "asymptotic",
        "invalid: ignores clustering and pairing",
    ),
]


def final_audit(macro: Callable[[str, str], None], gen: Path) -> None:
    tb = _load("audit_taubench.json")
    rows = []
    for key, label, est, nature, lim in TAU_METHODS:
        from agentaudit.compare import holm as _holm

        ps = [tb[dom]["rows"][key]["p"] for dom in ("retail", "airline")]
        adj = _holm(ps)
        cells = [f"{_p(p)} ({_p(float(a))})" for p, a in zip(ps, adj, strict=True)]
        rows.append(f"{label} & {est} & {nature} & {cells[0]} & {cells[1]} & {lim} \\\\")
    head = (
        "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{2.35cm}"
        ">{\\raggedright\\arraybackslash}p{3.0cm}>{\\raggedright\\arraybackslash}p{1.7cm}"
        "rr>{\\raggedright\\arraybackslash}p{2.6cm}}\n\\toprule\n"
        "Method & Estimand / null & Type & Retail $p$ & Airline $p$ "
        "& Limitation \\\\\n\\midrule\n"
    )
    (gen / "tab_tau_sensitivity.tex").write_text(
        head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    ph = tb["primary_holm_studentised_sign_flip"]
    macro("tauPrimaryRetailP", _p(tb["retail"]["rows"]["studentised_sign_flip"]["p"]))
    macro("tauPrimaryRetailHolm", _p(ph["retail"]))
    macro("tauPrimaryAirlineP", _p(tb["airline"]["rows"]["studentised_sign_flip"]["p"]))
    macro("tauRetailMajTies", _p(tb["retail"]["rows"]["mcnemar_majority_ties_dropped"]["p"]))
    macro("tauRetailPassHat", _p(tb["retail"]["rows"]["mcnemar_pass_hat4"]["p"]))
    macro("tauRetailPassAt", _p(tb["retail"]["rows"]["mcnemar_pass_at4"]["p"]))
    macro(
        "tauRetailMajEst",
        f"{100 * tb['retail']['rows']['mcnemar_majority_tie_as_fail']['est']:.1f}",
    )
    macro("tauRetailLabelP", f"{tb['retail']['rows']['label_permutation_within_task']['p']:.5f}")
    macro("tauRetailBootHi", f"{100 * tb['retail']['rows']['task_bootstrap_percentile']['hi']:.1f}")
    macro(
        "tauAirlineBootLo", f"{100 * tb['airline']['rows']['task_bootstrap_percentile']['lo']:.1f}"
    )
    macro(
        "tauAirlineBootHi", f"{100 * tb['airline']['rows']['task_bootstrap_percentile']['hi']:.1f}"
    )
    macro(
        "tauRetailNTies",
        str(tb["retail"]["rows"]["mcnemar_majority_ties_dropped"]["n_ties_dropped"]),
    )

    sw = _load("audit_swebench.json")
    sub = sw["subsets"]
    labels = {
        "official": "All entries, unobserved = unresolved (official score)",
        "complete_logs": "Entries with no missing logs",
        "complete_logs+single": "No missing logs, one attempt declared",
        "s3_cov+complete_logs": "Public log folders cover 500 and no missing logs",
    }
    rows = []
    for k, lab in labels.items():
        s = sub[k]
        ad = s["adjacent_pairs"]
        rows.append(
            f"{lab} & {s['n_agents']} & {s['n_pairs']:,} & {100 * s['frac_distinguishable_holm']:.1f} & "
            f"{ad['n_distinguishable_holm_all_pairs']}/{ad['n']} & {ad['n_distinguishable_raw']}/{ad['n']} \\\\".replace(
                ",", "\\,"
            )
        )
    for k, lab in (
        ("pairwise_complete_all129", "All 129, pairwise complete cases"),
        ("robust_bound_all129", "All 129, least favourable filling of missing logs"),
    ):
        s = sub[k]
        rows.append(
            f"{lab} & 129 & {s['n_pairs']:,} & {100 * s['frac_distinguishable_holm']:.1f} & -- & -- \\\\".replace(
                ",", "\\,"
            )
        )
    head = (
        "\\begin{tabular}{p{5.6cm}rrrrr}\n\\toprule\n"
        "Analysis & Entries & Pairs & Separable (\\%, Holm) & Adjacent, Holm & Adjacent, raw \\\\\n\\midrule\n"
    )
    (gen / "tab_swe_missing.tex").write_text(
        head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )
    macro("sweNNoLogs", str(sw["entries_with_no_logs"]))
    macro("sweMaxNoLogs", str(max(sw["no_logs_counts_nonzero"])))
    macro("sweCompleteN", str(sub["complete_logs"]["n_agents"]))
    macro("sweCompleteFracHolm", _pct(sub["complete_logs"]["frac_distinguishable_holm"]))
    macro("sweRobustFracHolm", _pct(sub["robust_bound_all129"]["frac_distinguishable_holm"]))
    macro("sweCCFracHolm", _pct(sub["pairwise_complete_all129"]["frac_distinguishable_holm"]))
    macro("sweNFolderCovWithNoLogs", str(len(sw["entries_s3_cov_but_with_no_logs"])))

    eq = _load("audit_equivalence.json")["cells"]
    rows = []
    for c in eq:
        r = c["rates"]
        lo, hi = c["wrong_decision_cp95"]
        rows.append(
            f"{c['margin'] * 100:.0f} & {c['delta'] * 100:+.0f} & {'in' if c['in_equivalence_region'] else 'out'} & "
            f"{100 * r['a_better']:.1f} & {100 * r['b_better']:.1f} & {100 * r['equivalent']:.1f} & "
            f"{100 * r['undecided']:.1f} & [{100 * lo:.2f}, {100 * hi:.2f}] & {100 * c['cs_miss_rate']:.2f} \\\\"
        )
    head = (
        "\\begin{tabular}{rrlrrrrcr}\n\\toprule\n"
        "Margin & True $\\Delta$ & Region & $a$ (\\%) & $b$ (\\%) & Equiv. (\\%) & Undec. (\\%) & "
        "Wrong decision, 95\\% CP (\\%) & CS miss (\\%) \\\\\n\\midrule\n"
    )
    (gen / "tab_equiv.tex").write_text(head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    macro("equivReps", str(eq[0]["reps"]))
    macro("equivWrongMaxUpper", f"{100 * max(c['wrong_decision_cp95'][1] for c in eq):.2f}\\%")
    macro("equivWrongMax", _pct(max(c["wrong_decision_rate"] for c in eq), 2))
    macro("equivMissMax", _pct(max(c["cs_miss_rate"] for c in eq), 2))
    out = [c for c in eq if not c["in_equivalence_region"] and c["margin"] > 0]
    macro("equivFalseEquivMax", _pct(max(c["rates"]["equivalent"] for c in out), 2))
    macro(
        "equivFalseEquivMaxUpper",
        f"{100 * max(c['false_equivalence_cp95'][1] for c in out):.2f}\\%",
    )
    it = sw["inclusion_threshold"]
    macro("sweNFiles", str(it["files_with_results_json"]))
    macro("sweNExcluded", str(len(it["excluded_below_50_resolved"])))

    # local-model study: sensitivity of the pair counts to the degenerate q2 control (stored p-values only)
    import numpy as np

    from agentaudit.compare import holm

    pw = _load("real_main.json")["analysis"]["pairwise"]
    sub_pw = [q for q in pw if "qwen2.5-3b-q2" not in (q["a"], q["b"])]
    h = holm(np.array([q["p_permutation"] for q in sub_pw]))
    macro("realNPairsNoQtwo", str(len(sub_pw)))
    macro("realNIndistNoQtwo", str(int((h >= 0.05).sum())))

    # Monte Carlo precision of every simulation file (worst case: a proportion near 1/2)
    def reps_in(o: Any) -> list[int]:
        if isinstance(o, dict):
            out = [o["reps"]] if isinstance(o.get("reps"), int) else []
            for v in o.values():
                out += reps_in(v)
            return out
        if isinstance(o, list):
            return [r for v in o for r in reps_in(v)]
        return []

    files = [
        ("sim_coverage.json", "S0 interval coverage"),
        ("sim_tests.json", "S0 type-I error"),
        ("sim_design.json", "S0 tasks vs repeats"),
        ("sim_ranking.json", "S0 ranking calibration"),
        ("sim_sequential.json", "S4 sequential"),
        ("sim_asym.json", "E1 skewed weak null"),
        ("sim_alloc.json", "E2 allocation"),
        ("sim_covrob.json", "E3 coverage robustness"),
        ("sim_seqm.json", "E5 sequential margins"),
        ("sim_rankpw.json", "E6 pairwise Holm"),
        ("audit_equivalence.json", "Audit A equivalence"),
    ]
    rows = []
    for fn, lab in files:
        reps = reps_in(_load(fn))
        if not reps:
            continue
        lo, hi = min(reps), max(reps)
        rr = f"{lo}" if lo == hi else f"{lo}--{hi}"
        rows.append(f"{lab} & {rr} & {100 * 0.5 / lo**0.5:.1f} \\\\")
    (gen / "tab_mcse.tex").write_text(
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "Simulation & Replications per cell & Worst-case MC s.e. (points) \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    cov = _load("sim_coverage.json")["mean_score"]
    wc = next(c for c in cov if c["tasks"] == 100 and c["runs"] == 10)
    pw_ = wc["coverage"]["wilson_pooled"]
    macro("covWilsonTenRunsSE", f"{100 * (pw_ * (1 - pw_) / wc['reps']) ** 0.5:.1f}")
    ty = _load("sim_tests.json")["cells"]
    top = next(
        c
        for c in ty
        if c["delta_theta"] == 0.0
        and c["tasks"] == 100
        and c["runs"] == 10
        and c["interaction_sd"] == 1.0
    )
    q = top["reject_rate"]["pooled_mcnemar"]
    macro("typeIPoolMcSE", f"{100 * (q * (1 - q) / top['reps']) ** 0.5:.1f}")

    from scipy import stats as _st

    sq = _load("sim_seqm.json")["cells"]
    outside_ = [c for c in sq if c["true_diff"] > c["margin"] > 0]
    worst = max(outside_, key=lambda c: c["rates"]["equivalent"])
    k = round(worst["rates"]["equivalent"] * worst["reps"])
    up = 1.0 if k == worst["reps"] else float(_st.beta.ppf(0.975, k + 1, worst["reps"] - k))
    macro("seqmFalseEquivUpper", _pct(up, 2))
    tie10 = next(c for c in eq if c["margin"] == 0.10 and c["delta"] == 0.0)
    macro("equivTieMTen", _pct(tie10["rates"]["equivalent"], 0))

    macro("tauRetailSkew", f"{tb['retail']['task_diff_skewness']:.2f}")
    macro("tauAirlineSkew", f"{tb['airline']['task_diff_skewness']:.2f}")

    bb = {c["mixing"]: c for c in _load("audit_bbmix.json")["cells"]}
    macro("bbTwoPointCov", _pct(bb["two_point"]["beta_binomial_profile_coverage"]))
    macro("bbBetaCov", _pct(bb["beta"]["beta_binomial_profile_coverage"]))
