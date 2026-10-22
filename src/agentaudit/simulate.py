"""Synthetic agents of known skill for validating statistics (no LLM involved).

Generative model (logit-normal item response model with agent-by-task interaction):

    p[m, t] = sigmoid(theta_m - b_t + g_mt),   b_t ~ N(0, sigma_b^2),  g_mt ~ N(0, sigma_g^2)
    Y[m, t, r] ~ Bernoulli(p[m, t])            independent across runs r.

Ground truth (population mean score, pass^k, pass@k, differences) is computed by
Gauss-Hermite quadrature, so interval coverage and test error rates can be measured
exactly against known truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from agentaudit.results import ResultSet

FloatArray = NDArray[np.float64]

_NODES, _WEIGHTS = np.polynomial.hermite_e.hermegauss(120)
_WEIGHTS = _WEIGHTS / _WEIGHTS.sum()


def _sigmoid(x: FloatArray) -> FloatArray:
    out: FloatArray = 1.0 / (1.0 + np.exp(-x))
    return out


@dataclass(frozen=True)
class SyntheticPopulation:
    """A population of tasks and a set of agents with fixed skills ``thetas``."""

    thetas: tuple[float, ...]
    sigma_task: float = 1.5  # spread of task difficulty
    sigma_interaction: float = 0.5  # agent-by-task interaction
    names: tuple[str, ...] | None = None

    @property
    def model_names(self) -> tuple[str, ...]:
        return self.names or tuple(f"agent{i}" for i in range(len(self.thetas)))

    def _sd(self) -> float:
        return float(np.hypot(self.sigma_task, self.sigma_interaction))

    def true_mean(self, m: int) -> float:
        """Population mean success probability of agent ``m`` over the task distribution."""
        return float(np.sum(_WEIGHTS * _sigmoid(self.thetas[m] + self._sd() * _NODES)))

    def true_pass_hat_k(self, m: int, k: int) -> float:
        return float(np.sum(_WEIGHTS * _sigmoid(self.thetas[m] + self._sd() * _NODES) ** k))

    def true_pass_at_k(self, m: int, k: int) -> float:
        p = _sigmoid(self.thetas[m] + self._sd() * _NODES)
        return float(np.sum(_WEIGHTS * (1 - (1 - p) ** k)))

    def true_diff(self, a: int, b: int) -> float:
        return self.true_mean(a) - self.true_mean(b)

    def sample_probs(self, n_tasks: int, rng: np.random.Generator) -> FloatArray:
        """Per-agent, per-task success probabilities, shape ``(agents, tasks)``."""
        theta = np.asarray(self.thetas)[:, None]
        b = rng.normal(0, self.sigma_task, (1, n_tasks))
        g = rng.normal(0, self.sigma_interaction, (len(self.thetas), n_tasks))
        return _sigmoid(theta - b + g)

    def sample(self, n_tasks: int, n_runs: int, rng: np.random.Generator | int = 0) -> ResultSet:
        gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        p = self.sample_probs(n_tasks, gen)
        scores = (gen.random((p.shape[0], n_tasks, n_runs)) < p[:, :, None]).astype(np.float64)
        return ResultSet.from_array(scores, self.model_names)
