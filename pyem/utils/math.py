
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit

def softmax(evs: ArrayLike, beta: float) -> np.ndarray:
    x = np.asarray(evs, dtype=float)
    x = x - np.max(x)  # stabilize
    p = np.exp(beta * x)
    return p / p.sum()

def norm2alpha(x: ArrayLike) -> np.ndarray:
    """Map R -> (0,1)."""
    return expit(np.asarray(x))

def alpha2norm(a: ArrayLike) -> np.ndarray:
    a = np.asarray(a)
    eps = 1e-12
    a = np.clip(a, eps, 1 - eps)
    return -np.log(1.0 / a - 1.0)

def norm2beta(x: ArrayLike, max_val: float = 20.0) -> np.ndarray:
    """Map R -> (0,max_val)."""
    return max_val / (1.0 + np.exp(-np.asarray(x)))

def beta2norm(b: ArrayLike, max_val: float = 20.0) -> np.ndarray:
    b = np.asarray(b)
    eps = 1e-12
    b = np.clip(b, eps, max_val - eps)
    return np.log(b / (max_val - b))

def check_bounds(val: float, lo: float, hi: float, penalty: float = 1e6) -> float | None:
    if val < lo or val > hi:
        return penalty
    return None
