import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from agentaudit.compare import (
    benjamini_hochberg,
    cohens_dz,
    compare,
    holm,
    mcnemar_exact,
    paired_bootstrap,
    pairwise,
    prob_superiority,
    sign_flip_permutation,
    task_diffs,
)
from agentaudit.results import ResultSet

pvals = st.lists(st.floats(1e-6, 1.0), min_size=1, max_size=30)


@given(st.integers(0, 40), st.integers(0, 40))
def test_mcnemar_matches_statsmodels(b: int, c: int) -> None:
    ref = mcnemar([[0, b], [c, 0]], exact=True).pvalue
    assert mcnemar_exact(b, c) == pytest.approx(float(ref), abs=1e-9)


@given(pvals)
def test_multiplicity_matches_statsmodels_and_dominates(p: list[float]) -> None:
    for fn, method in ((holm, "holm"), (benjamini_hochberg, "fdr_bh")):
        ref = multipletests(p, method=method)[1]
        np.testing.assert_allclose(fn(p), ref, atol=1e-9)
        assert np.all(fn(p) >= np.array(p) - 1e-12)
    assert np.all(holm(p) >= benjamini_hochberg(p) - 1e-12)


def test_permutation_exact_matches_enumeration_and_ttest_direction() -> None:
    d = np.array([0.2, 0.5, -0.1, 0.3, 0.4, 0.1, 0.25, 0.35])
    exact = sign_flip_permutation(d)
    mc = sign_flip_permutation(d, exact_below=0, n_perm=40_000, seed=1)
    assert exact == pytest.approx(mc, abs=0.01)
    assert exact == pytest.approx(stats.ttest_1samp(d, 0).pvalue, abs=0.03)
    assert sign_flip_permutation(np.zeros(5)) == 1.0


def test_type_i_error_is_controlled_under_null() -> None:
    rng = np.random.default_rng(0)
    rejections = {"perm": 0, "boot": 0, "mc": 0}
    reps = 400
    for i in range(reps):
        difficulty = rng.beta(1.2, 1.2, size=50)
        a = (rng.random((50, 4)) < difficulty[:, None]).astype(float)
        b = (rng.random((50, 4)) < difficulty[:, None]).astype(float)
        c = compare(a, b, n_boot=400, seed=i)
        rejections["perm"] += c.p_permutation < 0.05
        rejections["boot"] += c.p_bootstrap < 0.05
        rejections["mc"] += c.p_mcnemar < 0.05
    assert rejections["perm"] / reps < 0.09
    assert rejections["boot"] / reps < 0.10
    assert rejections["mc"] / reps < 0.09


def test_power_against_real_difference() -> None:
    rng = np.random.default_rng(1)
    hits = 0
    for i in range(100):
        base = rng.beta(2, 2, size=100)
        a = (rng.random((100, 3)) < np.clip(base + 0.1, 0, 1)[:, None]).astype(float)
        b = (rng.random((100, 3)) < base[:, None]).astype(float)
        c = compare(a, b, n_boot=300, seed=i)
        hits += c.p_permutation < 0.05 and c.diff > 0
    assert hits >= 60


def test_bootstrap_interval_and_effect_sizes() -> None:
    d = np.array([0.1, 0.3, 0.2, -0.05, 0.25, 0.15] * 5)
    lo, hi, p = paired_bootstrap(d, n_boot=4000)
    assert lo < d.mean() < hi and lo > 0 and p < 0.01
    assert prob_superiority([1, -1, 0, 0]) == 0.5
    assert np.isnan(cohens_dz([0.2, 0.2, 0.2]))
    assert cohens_dz(d) == pytest.approx(d.mean() / d.std(ddof=1))
    with pytest.raises(ValueError):
        task_diffs(np.zeros((3, 2)), np.zeros((4, 2)))


def test_pairwise_adjusts_and_validates() -> None:
    rng = np.random.default_rng(3)
    skill = np.array([0.3, 0.5, 0.5, 0.8])
    scores = (rng.random((4, 80, 4)) < skill[:, None, None]).astype(float)
    rs = ResultSet.from_array(scores, ["a", "b", "c", "d"])
    res = pairwise(rs, n_boot=500)
    assert len(res) == 6
    assert all(c.p_adjusted >= c.p_permutation - 1e-12 for c in res)
    bc = next(c for c in res if (c.a, c.b) == ("b", "c"))
    assert not bc.significant_ci and bc.p_adjusted > 0.05
    ad = next(c for c in res if (c.a, c.b) == ("a", "d"))
    assert ad.significant_ci and ad.diff < 0
    assert pairwise(rs.select(["a"])) == []
    for bad in ({"test": "x"}, {"correction": "x"}):
        with pytest.raises(ValueError):
            pairwise(rs, **bad)  # type: ignore[arg-type]
    for corr in ("bh", "none"):
        for test in ("bootstrap", "mcnemar"):
            assert len(pairwise(rs, correction=corr, test=test, n_boot=200)) == 6
