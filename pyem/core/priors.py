from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np

class Prior(Protocol):
    """Protocol for prior objects."""
    def logpdf(self, x: np.ndarray) -> float:
        """Return the log probability density of ``x``."""

@dataclass
class GaussianPrior:
    """Independent Gaussian prior."""

    mu: np.ndarray
    sigma: np.ndarray

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=float).reshape(-1)
        self.sigma = np.asarray(self.sigma, dtype=float).reshape(-1)
        if np.any(self.sigma <= 0):
            raise ValueError("sigma must be positive")

    def logpdf(self, x: np.ndarray) -> float:
        x = np.asarray(x).reshape(-1)
        var = self.sigma
        return float(-0.5 * np.sum(np.log(2 * np.pi * var) + (x - self.mu) ** 2 / var))

def default_prior(nparams: int, seed: int | None = None) -> GaussianPrior:
    rng = np.random.default_rng(seed)
    mu = 0.1 * rng.standard_normal(nparams)
    sigma = np.full(nparams, 100.0)
    return GaussianPrior(mu=mu, sigma=sigma)
