
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


def calc_fval(negll: float, params: ArrayLike, prior=None, output: str = 'npl') -> float:
    """Return objective value given a negative log-likelihood.

    Parameters
    ----------
    negll : float
        Negative log-likelihood of the data under the model.
    params : array-like
        Parameter vector passed to the prior.  Only used when `prior` is
        provided and ``output`` is ``"npl"``.
    prior : object or None, optional
        Object with a ``logpdf`` method returning the log prior density.
        When ``None`` the function reduces to returning ``negll``.
    output : {"npl", "nll"}
        Indicates whether to return the negative posterior likelihood
        (``"npl"``) or just the negative log-likelihood (``"nll"``).

    Returns
    -------
    float
        Objective value suitable for minimisation.  If the prior results in
        ``inf`` the value is capped at a large constant to keep optimisers
        stable.
    """
    if output == 'npl' and prior is not None and hasattr(prior, 'logpdf'):
        # Want to minimise -log[ P(data|h) * P(h) ]
        fval = -(-negll + prior.logpdf(np.asarray(params)))
        if np.isinf(fval):
            # Return a very large value so gradient-based optimisers keep going
            fval = 1e7
        return fval
    else:
        return negll
