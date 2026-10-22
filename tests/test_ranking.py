import numpy as np
import pytest

from agentaudit.ranking import (
    fraction_indistinguishable,
    indistinguishable_matrix,
    rank_uncertainty,
    split_half_rank_correlation,
)
from agentaudit.results import ResultSet


def make(skills: list[float], t: int = 120, r: int = 4, seed: int = 0) -> ResultSet:
    rng = np.random.default_rng(seed)
    base = rng.beta(2, 2, size=t)
    logit = np.log(base / (1 - base))
    probs = []
    for s in skills:
        shifted = 1 / (1 + np.exp(-(logit + s)))
        probs.append(shifted)
    p = np.array(probs)[:, :, None]
    return ResultSet.from_array((rng.random((len(skills), t, r)) < p).astype(float))


def test_rank_distribution_is_well_formed() -> None:
    ru = rank_uncertainty(make([0, 0.1, 1.5]), n_boot=800)
    np.testing.assert_allclose(ru.rank_probs.sum(axis=0), 1.0, atol=1e-9)
    np.testing.assert_allclose(ru.rank_probs.sum(axis=1), 1.0, atol=1e-9)
    assert ru.prob_top_k(3).tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert ru.rank_probs[2, 0] > 0.99  # clearly best
    pb = ru.prob_better
    np.testing.assert_allclose(pb + pb.T, 1.0, atol=1e-9)
    assert ru.table()[0]["model"] == "model2"
    assert ru.rank_lo[2] == 1.0 and ru.rank_hi[0] >= ru.rank_lo[0]


def test_close_agents_have_uncertain_ranks() -> None:
    ru = rank_uncertainty(make([0, 0.02, 0.04], t=60, r=2), n_boot=800)
    assert ru.rank_probs.max() < 0.8
    assert (ru.rank_hi - ru.rank_lo).max() >= 1


def test_indistinguishable_matrix_structure() -> None:
    rs = make([0, 0.0, 2.0])
    mat = indistinguishable_matrix(rs)
    assert (mat == mat.T).all() and mat.diagonal().all()
    assert mat[0, 1] and not mat[0, 2] and not mat[1, 2]
    assert fraction_indistinguishable(rs) == pytest.approx(1 / 3)


def test_split_half_reliability_tracks_separation() -> None:
    wide, _ = split_half_rank_correlation(make([0, 1, 2, 3], t=200), n_splits=50)
    narrow, _ = split_half_rank_correlation(make([0, 0.01, 0.02, 0.03], t=40, r=1), n_splits=50)
    assert wide > 0.8 and narrow < wide


def test_validation_and_degenerate() -> None:
    with pytest.raises(ValueError):
        rank_uncertainty(make([0.0]))
    one_task = ResultSet.from_array(np.ones((2, 1, 2)))
    assert np.isnan(split_half_rank_correlation(one_task)[0])
