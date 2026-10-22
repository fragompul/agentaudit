import json

import numpy as np

from agentaudit.analysis import analyze
from agentaudit.simulate import SyntheticPopulation


def test_analyze_balanced_is_json_serialisable_and_sane() -> None:
    pop = SyntheticPopulation((0.0, 0.1, 1.5), names=("a", "b", "c"))
    out = analyze(pop.sample(80, 5, 1), n_boot=300)
    json.dumps(out)
    assert out["balanced"] and len(out["models"]) == 3
    c = next(m for m in out["models"] if m["model"] == "c")
    assert c["lo"] < c["score"] < c["hi"]
    assert out["ranking"]["table"][0]["model"] == "c"
    assert 0 <= out["fraction_indistinguishable"] <= 1
    assert out["variance"]["g_coefficient"] > 0.5
    assert out["power"]["plan"][0]["n_runs"] == 1
    k5 = c["pass_hat_k"][5]
    assert k5[0] <= c["pass_at_k"][5][0]


def test_analyze_handles_missing_attempts_and_single_model() -> None:
    rs = SyntheticPopulation((0.0, 0.5)).sample(40, 4, 2)
    scores = rs.scores.copy()
    scores[0, :5, 3] = np.nan
    scores[1, 7, :2] = np.nan
    out = analyze(type(rs)(rs.models, rs.tasks, scores), n_boot=200)
    assert not out["balanced"] and out["balanced_tasks"] < 40
    one = analyze(rs.select([rs.models[0]]), n_boot=200)
    assert "ranking" not in one and one["models"][0]["task_icc"] is not None
    single_run = analyze(SyntheticPopulation((0.0, 1.0)).sample(30, 1, 3), n_boot=200)
    assert "variance" not in single_run and "ranking" in single_run
