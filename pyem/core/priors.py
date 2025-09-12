"""Prior distributions for hierarchical models."""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma
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


# ---- Additional prior families ----


@dataclass
class LaplacePrior:
    mu: np.ndarray
    b: np.ndarray  # scale > 0

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=float).reshape(-1)
        self.b = np.asarray(self.b, dtype=float).reshape(-1)
        if np.any(self.b <= 0):
            raise ValueError("b must be positive")

    def logpdf(self, x: np.ndarray) -> float:
        x = np.asarray(x).reshape(-1)
        return float(-np.sum(np.log(2 * self.b) + np.abs(x - self.mu) / self.b))


@dataclass
class StudentTPrior:
    mu: np.ndarray
    sigma: np.ndarray  # scale^2
    df: float = 5.0

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=float).reshape(-1)
        self.sigma = np.asarray(self.sigma, dtype=float).reshape(-1)
        if self.df <= 0 or np.any(self.sigma <= 0):
            raise ValueError("df>0 and sigma>0 required")

    def logpdf(self, x: np.ndarray) -> float:
        x = np.asarray(x).reshape(-1)
        v = self.df
        s2 = self.sigma
        return float(
            np.sum(
                lgamma((v + 1) / 2)
                - lgamma(v / 2)
                - 0.5 * np.log(v * np.pi * s2)
                - ((v + 1) / 2) * np.log(1 + ((x - self.mu) ** 2) / (v * s2))
            )
        )


@dataclass
class LogNormalPrior:
    mu_log: np.ndarray
    sigma_log: np.ndarray

    def __post_init__(self) -> None:
        self.mu_log = np.asarray(self.mu_log, dtype=float).reshape(-1)
        self.sigma_log = np.asarray(self.sigma_log, dtype=float).reshape(-1)
        if np.any(self.sigma_log <= 0):
            raise ValueError("sigma_log must be positive")

    def logpdf(self, x: np.ndarray) -> float:
        x = np.asarray(x).reshape(-1)
        if np.any(x <= 0):
            return -np.inf
        return float(
            -0.5
            * np.sum(((np.log(x) - self.mu_log) ** 2) / self.sigma_log + np.log(2 * np.pi * self.sigma_log))
            - np.sum(np.log(x))
        )


@dataclass
class UniformPrior:
    """Independent uniform prior with finite bounds."""

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, dtype=float).reshape(-1)
        self.upper = np.asarray(self.upper, dtype=float).reshape(-1)
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape")

    def logpdf(self, x: np.ndarray) -> float:
        x = np.asarray(x).reshape(-1)
        if np.any((x < self.lower) | (x > self.upper)):
            return -np.inf
        width = self.upper - self.lower
        return float(-np.sum(np.log(width)))

