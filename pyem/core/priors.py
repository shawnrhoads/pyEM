
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Protocol

class Prior(Protocol):
    def logpdf(self, x: np.ndarray) -> float: ...

@dataclass
class GaussianPrior:
    mu: np.ndarray      # shape (nparams,)
    sigma: np.ndarray   # variances, shape (nparams,)

    def __post_init__(self):
        self.mu = np.asarray(self.mu, dtype=float).reshape(-1)
        self.sigma = np.asarray(self.sigma, dtype=float).reshape(-1)
        if np.any(self.sigma <= 0):
            raise ValueError("sigma must be positive")

    def logpdf(self, x: np.ndarray) -> float:
        x = np.asarray(x).reshape(-1)
        var = self.sigma
        return float(-0.5 * np.sum(np.log(2*np.pi*var) + (x - self.mu)**2 / var))

def default_prior(nparams: int, seed: int | None = None) -> GaussianPrior:
    rng = np.random.default_rng(seed)
    mu = 0.1 * rng.standard_normal(nparams)
    sigma = np.full(nparams, 100.0)
    return GaussianPrior(mu=mu, sigma=sigma)



# ---- Additional prior families ----

from math import lgamma

@dataclass
class LaplacePrior:
    mu: np.ndarray
    b: np.ndarray  # scale > 0
    def __post_init__(self):
        self.mu = np.asarray(self.mu, dtype=float).reshape(-1)
        self.b = np.asarray(self.b, dtype=float).reshape(-1)
        if np.any(self.b <= 0):
            raise ValueError("b must be positive")
    def logpdf(self, x: np.ndarray) -> float:
        x = np.asarray(x).reshape(-1)
        return float(-np.sum(np.log(2*self.b) + np.abs(x - self.mu)/self.b))

@dataclass
class StudentTPrior:
    mu: np.ndarray
    sigma: np.ndarray  # scale^2
    df: float = 5.0
    def __post_init__(self):
        self.mu = np.asarray(self.mu, dtype=float).reshape(-1)
        self.sigma = np.asarray(self.sigma, dtype=float).reshape(-1)
        if self.df <= 0 or np.any(self.sigma <= 0):
            raise ValueError("df>0 and sigma>0 required")
    def logpdf(self, x: np.ndarray) -> float:
        x = np.asarray(x).reshape(-1)
        v = self.df
        s2 = self.sigma
        # independent t's
        return float(np.sum(
            lgamma((v+1)/2) - lgamma(v/2) - 0.5*np.log(v*np.pi*s2) - ((v+1)/2)*np.log(1 + ((x-self.mu)**2)/(v*s2))
        ))

@dataclass
class LogNormalPrior:
    mu_log: np.ndarray
    sigma_log: np.ndarray
    def __post_init__(self):
        self.mu_log = np.asarray(self.mu_log, dtype=float).reshape(-1)
        self.sigma_log = np.asarray(self.sigma_log, dtype=float).reshape(-1)
        if np.any(self.sigma_log <= 0):
            raise ValueError("sigma_log must be positive")
    def logpdf(self, x: np.ndarray) -> float:
        x = np.asarray(x).reshape(-1)
        if np.any(x <= 0):
            return -np.inf
        return float(-0.5*np.sum(((np.log(x)-self.mu_log)**2)/self.sigma_log + np.log(2*np.pi*self.sigma_log)) - np.sum(np.log(x)))
