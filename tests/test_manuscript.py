"""Consistency between the manuscript, its generated macros and the stored results.

These tests re-run no experiment: they check the paper's numbers against ``results/``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "latex" / "generated"
SECTIONS = ROOT / "latex" / "sections"

pytestmark = pytest.mark.skipif(not GEN.exists(), reason="latex sources not present")


def _macros() -> dict[str, str]:
    out: dict[str, str] = {}
    for f in ("macros.tex", "real_macros.tex"):
        text = (GEN / f).read_text(encoding="utf-8")
        for m in re.finditer(r"\\(?:re)?newcommand\{\\(\w+)\}\{(.*)\}\s*$", text, re.M):
            out[m.group(1)] = m.group(2)
    return out


def _results(name: str) -> dict:  # type: ignore[type-arg]
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def _pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}\\%"


def test_every_macro_used_in_the_text_is_defined() -> None:
    defined = set(_macros())
    main = (ROOT / "latex" / "main.tex").read_text(encoding="utf-8")
    defined |= set(re.findall(r"\\newcommand\{\\(\w+)\}", main))
    text = main + "".join(p.read_text(encoding="utf-8") for p in SECTIONS.glob("*.tex"))
    # restrict to the prefixes of generated macros so that LaTeX built-ins are ignored
    prefixes = (
        "cov",
        "typeI",
        "asym",
        "alloc",
        "design",
        "seq",
        "rank",
        "swe",
        "tau",
        "real",
        "abl",
    )
    used = {m for m in re.findall(r"\\([A-Za-z]+)\{\}", text) if m.startswith(prefixes)}
    used |= {m for m in re.findall(r"\\([A-Za-z]+)(?![A-Za-z])", text) if m.startswith(prefixes)}
    ignore = {"seqBudget"} - defined
    missing = sorted(
        m for m in used if m not in defined and m not in ignore and m not in {"cov", "tau"}
    )
    assert not missing, missing


def test_headline_simulation_macros_match_results() -> None:
    m = _macros()
    cov = {
        (c["tasks"], c["runs"]): c["coverage"] for c in _results("sim_coverage.json")["mean_score"]
    }
    assert m["covWilsonTenRuns"] == _pct(cov[(100, 10)]["wilson_pooled"])
    assert m["covBootTenRuns"] == _pct(cov[(100, 10)]["bootstrap_tasks"])
    tests = _results("sim_tests.json")["cells"]
    top = next(
        c
        for c in tests
        if c["delta_theta"] == 0.0
        and c["tasks"] == 100
        and c["runs"] == 10
        and c["interaction_sd"] == 1.0
    )["reject_rate"]
    assert m["typeIPoolMc"] == _pct(top["pooled_mcnemar"])
    assert m["typeIPerm"] == _pct(top["task_permutation"])


def test_extra_experiment_macros_match_results() -> None:
    m = _macros()
    asym = _results("sim_asym.json")["cells"]
    worst = max(max(c["reject_rate"].values()) for c in asym)
    assert m["asymMaxAny"] == _pct(worst)
    rank = _results("sim_rankpw.json")["cells"]
    assert m["rankpwFWE"] == _pct(max(c["false_rejection_rate_equal_pair"] for c in rank))
    seq = _results("sim_seqm.json")["cells"]
    assert all(c["rates"]["equivalent"] == 0 for c in seq if c["margin"] == 0)


def test_swebench_macros_and_coverage_audit_are_consistent() -> None:
    m = _macros()
    d = _results("reanalysis_swebench.json")
    assert m["sweNAgents"] == str(d["n_agents"])
    assert m["sweFracHolm"] == _pct(d["frac_distinguishable_holm"])
    audit = d["coverage_audit"]
    assert (
        audit["coverage_verified"]
        + audit["coverage_unverifiable_no_public_logs"]
        + audit["coverage_incomplete"]
        == d["n_agents"]
    )
    assert (
        audit["attempts_1"] + audit["attempts_multiple"] + audit["attempts_unknown"]
        == d["n_agents"]
    )
    for key, sub in d["subsets"].items():
        n = sub["n_agents"]
        assert sub["n_pairs"] == n * (n - 1) // 2, key
        assert sub["adjacent_pairs"]["n"] == n - 1, key


def test_taubench_macros_match_results() -> None:
    m = _macros()
    d = _results("reanalysis_taubench.json")
    assert m["tauRetailPPerm"] == f"{d['retail']['task_level_comparison']['p_permutation']:.3f}"
    assert m["tauAirlineTasks"] == str(d["airline"]["n_tasks"])
    assert (
        d["holm_p_permutation_two_domains"]["retail"]
        >= d["retail"]["task_level_comparison"]["p_permutation"]
    )


def test_swebench_sensitivity_table_is_part_of_the_paper() -> None:
    text = (SECTIONS / "results.tex").read_text(encoding="utf-8")
    assert "tab_swe_robust" in text and "tab:swerobust" in text
    assert (GEN / "tab_swe_robust.tex").exists()


def test_no_green_or_red_link_boxes() -> None:
    main = (ROOT / "latex" / "main.tex").read_text(encoding="utf-8")
    assert "hidelinks" in main
