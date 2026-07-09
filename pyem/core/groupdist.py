"""Group-level M-step families for empirical-Bayes EM.

Each "group distribution" family provides three methods:

- ``update(m, inv_h) -> hyper``: the empirical-Bayes M-step given subject MAP
  means ``m`` (shape ``(nparams, nsubjects)``) and per-subject inverse-Hessians
  ``inv_h`` (shape ``(nparams, nparams, nsubjects)``).
- ``make_prior(hyper) -> Prior``: build the per-iteration E-step regularizer
  (a prior object with ``.logpdf``) from the current hyperparameters.
- ``moments(hyper) -> (mu, sigma, flag)``: return group mean, group variance
  (per parameter), and an int ``flag`` (1 good / 0 bad) for the EM loop's
  posterior / convergence bookkeeping.

A factory ``make_group(name, df=8.0)`` dispatches by name.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erf

from .priors import GaussianPrior, LaplacePrior, StudentTPrior, CauchyPrior


class GaussianGroup:
    """Gaussian group distribution (the classic MFX M-step)."""

    def update(self, m: np.ndarray, inv_h: np.ndarray):
        # Bit-identical to em._calc_group_gaussian.
        nsub = m.shape[1]
        npar = m.shape[0]
        mu = np.mean(m, axis=1)
        sigma = np.zeros(npar)
        for s in range(nsub):
            sigma += m[:, s] ** 2 + np.diag(inv_h[:, :, s])
        sigma = sigma / nsub - mu ** 2
        flag = 1
        if np.min(sigma) < 0:
            flag = 0
        return mu, sigma, flag

    def moments(self, hyper):
        return hyper

    def make_prior(self, hyper):
        mu, sigma, _flag = hyper
        return GaussianPrior(mu=mu, sigma=sigma)


def _folded_normal_abs_dev(d: np.ndarray, s: np.ndarray) -> np.ndarray:
    """E|theta - mu| for theta ~ N(mi, si^2), with d = mi - mu, s = si.

    E|theta-mu| = s*sqrt(2/pi)*exp(-d^2/(2 s^2)) + d*erf(d/(s*sqrt(2)))
    """
    s = np.maximum(s, 1e-12)
    return (s * np.sqrt(2.0 / np.pi) * np.exp(-(d ** 2) / (2.0 * s ** 2))
            + d * erf(d / (s * np.sqrt(2.0))))


class LaplaceGroup:
    """Laplace group distribution."""

    def update(self, m: np.ndarray, inv_h: np.ndarray):
        npar = m.shape[0]
        loc = np.zeros(npar)
        scale = np.zeros(npar)
        n_failed = 0
        for p in range(npar):
            mp = m[p, :]
            sp = np.sqrt(np.maximum(inv_h[p, p, :], 0.0))

            def objective(mu, mp=mp, sp=sp):
                return float(np.sum(_folded_normal_abs_dev(mp - mu, sp)))

            res = minimize_scalar(objective, method="brent")
            if not res.success:
                n_failed += 1
            mu_star = float(res.x)
            b = float(np.mean(_folded_normal_abs_dev(mp - mu_star, sp)))
            loc[p] = mu_star
            scale[p] = max(b, 1e-12)
        if n_failed:
            warnings.warn(
                f"LaplaceGroup M-step: minimize_scalar failed to converge "
                f"for {n_failed} of {npar} parameter(s)",
                RuntimeWarning,
                stacklevel=2,
            )
        return {"loc": loc, "scale": scale}

    def moments(self, hyper):
        loc = np.asarray(hyper["loc"], dtype=float)
        scale = np.asarray(hyper["scale"], dtype=float)
        sigma = 2.0 * scale ** 2
        flag = int(np.all(np.isfinite(sigma)) and np.all(sigma > 0))
        return loc, sigma, flag

    def make_prior(self, hyper):
        return LaplacePrior(loc=hyper["loc"], scale=hyper["scale"])


class StudentTGroup:
    """Student-t group distribution via a scale-mixture IRLS EM."""

    def __init__(self, df: float = 8.0):
        self.df = float(df)

    def update(self, m: np.ndarray, inv_h: np.ndarray):
        df = self.df
        npar = m.shape[0]
        loc = np.zeros(npar)
        scale = np.zeros(npar)
        n_not_converged = 0
        for p in range(npar):
            mi = m[p, :]
            si2 = np.maximum(inv_h[p, p, :], 0.0)
            mu = float(np.median(mi))
            sigma2 = float(np.median(np.abs(mi - mu)) ** 2 * 2.198 + 1e-6)
            converged = False
            for _ in range(100):
                mu_old = mu
                sigma2_old = sigma2
                w = (df + 1.0) / (df + ((mi - mu) ** 2 + si2) / sigma2)
                sw = np.sum(w)
                mu = float(np.sum(w * mi) / sw)
                sigma2 = float(np.sum(w * ((mi - mu) ** 2 + si2)) / sw)
                sigma2 = max(sigma2, 1e-12)
                if abs(mu - mu_old) < 1e-8 and abs(sigma2 - sigma2_old) < 1e-10:
                    converged = True
                    break
            if not converged:
                n_not_converged += 1
            loc[p] = mu
            scale[p] = np.sqrt(sigma2)
        if n_not_converged:
            warnings.warn(
                f"StudentTGroup M-step (df={df}): IRLS did not converge within "
                f"100 iterations for {n_not_converged} of {npar} parameter(s)",
                RuntimeWarning,
                stacklevel=2,
            )
        return {"loc": loc, "scale": scale, "df": df}

    def moments(self, hyper):
        loc = np.asarray(hyper["loc"], dtype=float)
        scale = np.asarray(hyper["scale"], dtype=float)
        df = float(hyper["df"])
        # Finite variance seed; reuse priors.StudentTPrior capping logic.
        _mu, var = StudentTPrior(loc=loc, scale=scale,
                                 df=np.full(loc.shape, df)).init_moments()
        flag = int(np.all(np.isfinite(var)) and np.all(var > 0))
        return loc, var, flag

    def make_prior(self, hyper):
        df = float(hyper["df"])
        if df > 1.0:
            return StudentTPrior(loc=hyper["loc"], scale=hyper["scale"],
                                 df=np.full(np.asarray(hyper["loc"]).reshape(-1).shape, df))
        return CauchyPrior(loc=hyper["loc"], scale=hyper["scale"])


class CauchyGroup(StudentTGroup):
    """Cauchy group distribution (Student-t with df=1)."""

    def __init__(self):
        super().__init__(df=1.0)


def make_group(name: str, df: float = 8.0):
    """Factory for group-distribution families.

    ``df`` is ignored when ``name == "cauchy"`` (Cauchy is df=1 by definition).
    """
    if name == "gaussian":
        return GaussianGroup()
    if name == "laplace":
        return LaplaceGroup()
    if name == "student_t":
        return StudentTGroup(df)
    if name == "cauchy":
        return CauchyGroup()
    raise ValueError(f"unknown mstep {name!r}")
