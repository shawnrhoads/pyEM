from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from scipy import stats

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

    def init_moments(self) -> tuple[np.ndarray, np.ndarray]:
        return self.mu.copy(), self.sigma.copy()

def default_prior(nparams: int, seed: int | None = None) -> GaussianPrior:
    rng = np.random.default_rng(seed)
    mu = 0.1 * rng.standard_normal(nparams)
    sigma = np.full(nparams, 100.0)
    return GaussianPrior(mu=mu, sigma=sigma)


@dataclass
class UniformPrior:
    """Independent bounded-uniform prior. logpdf = -sum(log(hi-lo)) inside the box, -inf outside."""
    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self) -> None:
        self.lo = np.asarray(self.lo, dtype=float).reshape(-1)
        self.hi = np.asarray(self.hi, dtype=float).reshape(-1)
        if np.any(self.hi <= self.lo):
            raise ValueError("hi must exceed lo")

    def logpdf(self, x: np.ndarray) -> float:
        x = np.asarray(x).reshape(-1)
        if np.any(x < self.lo) or np.any(x > self.hi):
            return -np.inf
        return float(-np.sum(np.log(self.hi - self.lo)))

    def init_moments(self) -> tuple[np.ndarray, np.ndarray]:
        mu = 0.5 * (self.lo + self.hi)
        var = (self.hi - self.lo) ** 2 / 12.0
        return mu, var


@dataclass
class LaplacePrior:
    """Independent Laplace prior. scale is the diversity b (std = b*sqrt(2))."""
    loc: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        self.loc = np.asarray(self.loc, dtype=float).reshape(-1)
        self.scale = np.asarray(self.scale, dtype=float).reshape(-1)
        if np.any(self.scale <= 0):
            raise ValueError("scale must be positive")

    def logpdf(self, x: np.ndarray) -> float:
        x = np.asarray(x).reshape(-1)
        return float(np.sum(stats.laplace.logpdf(x, loc=self.loc, scale=self.scale)))

    def init_moments(self) -> tuple[np.ndarray, np.ndarray]:
        return self.loc.copy(), 2.0 * self.scale ** 2


@dataclass
class StudentTPrior:
    """Independent Student-t prior. scale is std-like; df = degrees of freedom."""
    loc: np.ndarray
    scale: np.ndarray
    df: np.ndarray

    def __post_init__(self) -> None:
        self.loc = np.asarray(self.loc, dtype=float).reshape(-1)
        self.scale = np.asarray(self.scale, dtype=float).reshape(-1)
        self.df = np.asarray(self.df, dtype=float).reshape(-1)
        if np.any(self.scale <= 0) or np.any(self.df <= 0):
            raise ValueError("scale and df must be positive")

    def logpdf(self, x: np.ndarray) -> float:
        x = np.asarray(x).reshape(-1)
        return float(np.sum(stats.t.logpdf(x, df=self.df, loc=self.loc, scale=self.scale)))

    def init_moments(self) -> tuple[np.ndarray, np.ndarray]:
        raw = np.where(self.df > 2.0,
                       self.scale ** 2 * self.df / np.clip(self.df - 2.0, 1e-6, None),
                       np.inf)
        var = np.minimum(raw, self.scale ** 2 * 100.0)
        return self.loc.copy(), var


@dataclass
class CauchyPrior:
    """Independent Cauchy prior (Student-t df=1). scale is half-width gamma.
    Cauchy has no finite variance; init_moments returns a large finite fallback."""
    loc: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        self.loc = np.asarray(self.loc, dtype=float).reshape(-1)
        self.scale = np.asarray(self.scale, dtype=float).reshape(-1)
        if np.any(self.scale <= 0):
            raise ValueError("scale must be positive")

    def logpdf(self, x: np.ndarray) -> float:
        x = np.asarray(x).reshape(-1)
        return float(np.sum(stats.cauchy.logpdf(x, loc=self.loc, scale=self.scale)))

    def init_moments(self) -> tuple[np.ndarray, np.ndarray]:
        return self.loc.copy(), self.scale ** 2 * 100.0


@dataclass
class IndependentPrior:
    """Per-parameter composer: one 1-D prior per coordinate."""
    priors: list

    def __post_init__(self) -> None:
        for i, p in enumerate(self.priors):
            if hasattr(p, "init_moments"):
                mu0 = np.atleast_1d(p.init_moments()[0])
                if mu0.size != 1:
                    raise ValueError(
                        f"IndependentPrior component {i} must be a 1-D (single-parameter) "
                        f"prior, but it spans {mu0.size} parameters"
                    )

    def logpdf(self, x: np.ndarray) -> float:
        x = np.asarray(x).reshape(-1)
        if len(x) != len(self.priors):
            raise ValueError("x length must match number of component priors")
        return float(np.sum([p.logpdf(x[j:j + 1]) for j, p in enumerate(self.priors)]))

    def init_moments(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.priors:
            return np.zeros(0), np.zeros(0)
        moments = [p.init_moments() for p in self.priors]
        mu = np.concatenate([np.atleast_1d(m[0]) for m in moments])
        var = np.concatenate([np.atleast_1d(m[1]) for m in moments])
        return mu, var
