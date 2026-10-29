"""Consistency of stored final-audit aggregates with the headline claims (aggregate-level check)."""

import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"


def _load(name: str) -> dict:
    return json.loads((R / name).read_text())


def test_taubench_headline_values() -> None:
    t = _load("audit_taubench.json")
    assert abs(t["retail"]["est_diff"] - 0.1022) < 1e-3
    assert t["retail"]["rows"]["studentised_sign_flip"]["p"] < 0.01
    assert t["primary_holm_studentised_sign_flip"]["retail"] < 0.05
    assert t["airline"]["rows"]["studentised_sign_flip"]["p"] > 0.4
    # the majority-vote McNemar is not significant, and it targets a different estimand
    assert t["retail"]["rows"]["mcnemar_majority_tie_as_fail"]["p"] > 0.05


def test_swebench_official_denominators() -> None:
    s = _load("audit_swebench.json")["subsets"]["official"]
    assert (s["n_agents"], s["n_pairs"]) == (129, 8256)
    assert s["adjacent_pairs"]["n_distinguishable_holm_all_pairs"] == 0


def test_equivalence_simulation_respects_alpha() -> None:
    e = _load("audit_equivalence.json")
    assert e["cells"][0]["reps"] >= 2000
    for c in e["cells"]:
        assert c["wrong_decision_rate"] <= 0.05
        assert c["cs_miss_rate"] <= 0.05
