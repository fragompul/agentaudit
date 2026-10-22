"""Write docs/results.md and the key-results block of README.md from results/*.json."""

from __future__ import annotations

import json
import re
from pathlib import Path

R = Path("results")


def load(name: str) -> dict | None:  # type: ignore[type-arg]
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def sim_section() -> list[str]:
    cov, tests, seq, des = (
        load(n)
        for n in ("sim_coverage.json", "sim_tests.json", "sim_sequential.json", "sim_design.json")
    )
    out = ["## Simulation: do the statistics work?", ""]
    if cov:
        pick = {(c["tasks"], c["runs"]): c["coverage"] for c in cov["mean_score"]}
        out += [
            "Coverage of nominal 95% intervals for the benchmark score (100 tasks, 2000 replications):",
            "",
            "| runs/task | pooled Wilson | task bootstrap | BCa | hierarchical beta-binomial |",
            "|---|---|---|---|---|",
        ]
        for r in (1, 3, 10):
            c = pick[(100, r)]
            out.append(
                f"| {r} | {pct(c['wilson_pooled'])} | {pct(c['bootstrap_tasks'])} | "
                f"{pct(c['bootstrap_tasks_bca'])} | {pct(c['beta_binomial_profile'])} |"
            )
        out += ["", "![coverage](assets/coverage.svg)", ""]
    if tests:
        c = next(
            x
            for x in tests["cells"]
            if x["delta_theta"] == 0
            and x["tasks"] == 100
            and x["runs"] == 10
            and x["interaction_sd"] == 1.0
        )
        rr = c["reject_rate"]
        out += [
            "Type-I error at alpha = 0.05 (equal-skill agents, 100 tasks, 10 runs, strong interaction): "
            f"pooled z-test {pct(rr['pooled_ztest'])}, pooled run-level McNemar {pct(rr['pooled_mcnemar'])}, "
            f"task sign-flip {pct(rr['task_permutation'])}, task bootstrap {pct(rr['task_bootstrap'])}.",
            "",
            "![type I](assets/typeI.svg)",
            "",
        ]
    if des:
        rows = [r for r in des["rows"] if r["budget"] == 1200]
        out += [
            "Fixed budget of 1200 attempts: power of the paired test by runs per task.",
            "",
            "| runs/task | tasks | power |",
            "|---|---|---|",
        ]
        out += [f"| {r['runs']} | {r['tasks']} | {pct(r['power'])} |" for r in rows]
        out += ["", "![design](assets/design.svg)", ""]
    if seq:
        n = seq["rows"][0]["decision_rate"]
        out += [
            f"Sequential testing under the null: naive peeking declares a winner in {pct(n['peek'])} of runs, "
            f"the confidence sequence in {pct(n['seq'])}, the fixed-n test in {pct(n['fixed'])}.",
            "",
            "![sequential](assets/sequential.svg)",
            "",
        ]
    return out


def public_section() -> list[str]:
    swe, tau = load("reanalysis_swebench.json"), load("reanalysis_taubench.json")
    out = ["## Public results", ""]
    if swe:
        out += [
            f"**SWE-bench Verified** ({swe['n_agents']} public submissions, {swe['n_pairs']} pairs, one run each): "
            f"{pct(swe['frac_distinguishable_holm'])} of pairs are distinguishable after Holm adjustment "
            f"({pct(swe['frac_distinguishable_raw'])} unadjusted); of the {swe['adjacent_pairs']['n']} adjacent leaderboard positions, "
            f"{pct(swe['adjacent_pairs']['frac_distinguishable_holm_within_adjacent'])} are distinguishable.",
            "",
            "![swe-bench](assets/swebench.svg)",
            "",
        ]
    if tau:
        out += [
            "**tau-bench** (gpt-4o vs claude-3.5-sonnet, 4 trials per task):",
            "",
            "| domain | tasks | diff (pts) | 95% CI | task-level p | pooled z p |",
            "|---|---|---|---|---|---|",
        ]
        for d in ("airline", "retail"):
            c, p = tau[d]["task_level_comparison"], tau[d]["pooled_shortcut"]
            out.append(
                f"| {d} | {tau[d]['n_tasks']} | {100 * c['diff']:+.1f} | [{100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f}] | "
                f"{c['p_permutation']:.3f} | {p['pooled_ztest_p']:.3f} |"
            )
        out += ["", "![tau-bench](assets/taubench.svg)", ""]
    return out


def real_section() -> list[str]:
    d = load("real_main.json")
    if not d:
        return []
    a = d["analysis"]
    out = [
        "## Local models on AgentAudit-Bench",
        "",
        f"{len(d['models'])} local models, {d['n_tasks']} core tasks, {d['n_runs']} runs per task, temperature 0.7.",
        "",
        "| model | score | 95% CI | pass^k (k=max) | pass@k (k=max) | P(best) |",
        "|---|---|---|---|---|---|",
    ]
    pbest = {r["model"]: r["p_best"] for r in a["ranking"]["table"]} if "ranking" in a else {}
    for m in sorted(a["models"], key=lambda m: -m["score"]):
        k = str(max(int(x) for x in m["pass_hat_k"]))
        out.append(
            f"| {m['model']} | {pct(m['score'])} | [{pct(m['lo'])}, {pct(m['hi'])}] | "
            f"{pct(m['pass_hat_k'][k][0])} (k={k}) | {pct(m['pass_at_k'][k][0])} | {pct(pbest.get(m['model'], 0), 0)} |"
        )
    if "variance" in a:
        s = a["variance"]["shares"]
        out += [
            "",
            f"Variance shares: agent skill {pct(s['model'])}, task difficulty {pct(s['task'])}, agent x task {pct(s['model_task'])}, "
            f"run noise {pct(s['run'])}. Pairs indistinguishable (Holm): {pct(a['fraction_indistinguishable'])}.",
        ]
    return [*out, ""]


def main() -> None:
    body = [
        "# Results",
        "",
        "Generated by `scripts/make_results_docs.py` from `results/*.json`.",
        "",
    ]
    body += sim_section() + public_section() + real_section()
    Path("docs/results.md").write_text("\n".join(body) + "\n")
    readme = Path("README.md")
    if readme.exists():
        txt = readme.read_text()
        block = "\n".join(public_section() + sim_section()[:0] + real_section()).replace(
            "](assets/", "](docs/assets/"
        )
        block = block.replace("## ", "### ")
        new = re.sub(
            r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->",
            lambda _m: f"<!-- RESULTS:START -->\n{block}\n<!-- RESULTS:END -->",
            txt,
            flags=re.S,
        )
        readme.write_text(new)
    print("wrote docs/results.md")


if __name__ == "__main__":
    main()
