"""Distil results/*.json into the compact payload that the landing page charts read.

Every figure on the site comes from here, so the page cannot drift from the paper.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def _load(name: str) -> Any:
    return json.loads((RES / f"{name}.json").read_text(encoding="utf-8"))


def _pretty(name: str) -> str:
    return re.sub(r"^\d{8}_", "", name).replace("_", " ")


def build() -> dict[str, Any]:
    sw = _load("reanalysis_swebench")
    tau = _load("reanalysis_taubench")
    real = _load("real_main")
    cov = _load("sim_coverage")
    seq = _load("sim_sequential")
    tests = _load("sim_tests")
    abl = _load("ablation_main")
    an = real["analysis"]

    null_cells = [c for c in tests["cells"] if c["true_diff"] == 0]
    return {
        "swebench": {
            "n_agents": sw["n_agents"],
            "n_tasks": sw["n_tasks"],
            "n_pairs": sw["n_pairs"],
            "frac_raw": sw["frac_distinguishable_raw"],
            "frac_holm": sw["frac_distinguishable_holm"],
            "adjacent_n": sw["adjacent_pairs"]["n"],
            "adjacent_holm": sw["adjacent_pairs"]["frac_distinguishable_holm_within_adjacent"],
            "score_range": sw["score_range"],
            "median_gap_needed": sw["median_gap_needed"],
            "by_gap": sw["by_score_gap"],
            "top": [
                {
                    "name": _pretty(t["name"]),
                    "score": t["score"],
                    "rank_lo": round(t["rank_lo"]),
                    "rank_hi": round(t["rank_hi"]),
                    "p_best": t["p_best"],
                }
                for t in sw["top10"]
            ],
        },
        "tau": {
            dom: {
                "tasks": tau[dom]["n_tasks"],
                "runs": tau[dom]["n_runs"],
                "diff": tau[dom]["task_level_comparison"]["diff"],
                "ci": tau[dom]["task_level_comparison"]["ci"],
                "p_task": tau[dom]["task_level_comparison"]["p_permutation"],
                "p_pooled": tau[dom]["pooled_shortcut"]["pooled_ztest_p"],
                "tasks_for_80": tau[dom]["tasks_needed_for_80pct_power_at_observed_gap"],
            }
            for dom in ("airline", "retail")
        },
        "local": {
            "n_tasks": an["n_tasks"],
            "n_runs": an["n_runs"],
            "frac_indist": an["fraction_indistinguishable"],
            "models": [
                {
                    "model": m["model"],
                    "score": m["score"],
                    "lo": m["lo"],
                    "hi": m["hi"],
                    "pass_hat_10": m["pass_hat_k"]["10"][0],
                    "pass_at_10": m["pass_at_k"]["10"][0],
                }
                for m in sorted(an["models"], key=lambda m: -m["score"])
            ],
            "shares": an["variance"]["shares"] if "variance" in an else None,
        },
        "coverage": {
            "nominal": cov["nominal"],
            "rows": [
                {
                    "tasks": r["tasks"],
                    "runs": r["runs"],
                    "pooled": r["coverage"]["wilson_pooled"],
                    "task_boot": r["coverage"]["bootstrap_tasks"],
                    "two_stage": r["coverage"]["bootstrap_two_stage"],
                    "profile": r["coverage"]["beta_binomial_profile"],
                }
                for r in cov["mean_score"]
            ],
        },
        "peeking": [
            {
                "true_diff": r["true_diff"],
                **{k: r["decision_rate"][k] for k in ("seq", "fixed", "peek")},
            }
            for r in seq["rows"]
        ],
        "type_i": {
            "worst_pooled": max(
                max(c["reject_rate"]["pooled_ztest"], c["reject_rate"]["pooled_mcnemar"])
                for c in null_cells
            ),
            "worst_task_level": max(c["reject_rate"]["task_permutation"] for c in null_cells),
        },
        "ablation": {
            "full_frac": abl["full"]["frac_pairs_distinguishable"],
            "rows": [
                {k: r[k] for k in ("tasks", "runs", "frac_pairs_distinguishable", "tau_mean")}
                for r in abl["subdesign"]
            ],
        },
    }


def write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(), separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    out = ROOT / "site" / "data" / "results.json"
    write(out)
    print(f"wrote {out}")
