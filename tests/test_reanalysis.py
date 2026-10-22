import numpy as np
import pytest

from agentaudit.compare import mcnemar_exact
from agentaudit.reanalysis import mcnemar_matrix, reanalyze_single_run


def board(skills: list[float], t: int = 300, seed: int = 0) -> np.ndarray:  # type: ignore[type-arg]
    rng = np.random.default_rng(seed)
    d = rng.normal(0, 1.2, t)
    return np.array([rng.random(t) < 1 / (1 + np.exp(-(s - d))) for s in skills])


def test_matrix_matches_scalar_mcnemar() -> None:
    x = board([0, 0.2, 1.0, 0.0])
    p, only = mcnemar_matrix(x)
    for i in range(4):
        for j in range(4):
            if i != j:
                assert p[i, j] == pytest.approx(mcnemar_exact(int(only[i, j]), int(only[j, i])))
    np.testing.assert_allclose(p, p.T)
    assert (np.diag(p) == 1).all()


def test_reanalysis_separates_close_and_far_pairs() -> None:
    skills = [0.0, 0.05, 0.1, 0.15, 2.0, 2.05]
    r = reanalyze_single_run(board(skills), [f"a{i}" for i in range(6)], n_boot=300)
    assert r["n_pairs"] == 15
    assert r["frac_distinguishable_holm"] <= r["frac_distinguishable_raw"] + 1e-12
    gaps = [g for g in r["by_score_gap"] if g["pairs"]]
    assert gaps[0]["frac_distinguishable_holm"] < gaps[-1]["frac_distinguishable_holm"]
    assert r["top10"][0]["name"] in {"a4", "a5"}
    assert 0 <= r["top10"][0]["rank_lo"] <= r["top10"][0]["rank_hi"]
    assert r["adjacent_pairs"]["n"] == 5


def test_identical_agents_are_all_indistinguishable() -> None:
    x = np.tile(board([0.0], 100), (4, 1))
    r = reanalyze_single_run(x, list("abcd"), n_boot=100)
    assert r["frac_distinguishable_raw"] == 0 and r["median_gap_needed"] is None
