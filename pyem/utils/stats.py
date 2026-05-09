
from __future__ import annotations
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import logsumexp
from scipy.stats import norm

def calc_LME(inv_h: np.ndarray, NPL: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    nparams = inv_h.shape[0]
    nsubjects = inv_h.shape[2]
    good = np.zeros(nsubjects)
    Lap = np.zeros(nsubjects)
    for i in range(nsubjects):
        try:
            sign, logdet = np.linalg.slogdet(inv_h[:, :, i])
            if sign <= 0:
                raise ValueError("non-posdef Hessian")
            Lap[i] = -NPL[i] - 0.5 * (-logdet) + (nparams/2)*np.log(2*np.pi)
            good[i] = 1
        except Exception:
            Lap[i] = np.nan
            good[i] = 0
    if np.all(np.isnan(Lap)):
        Lap[:] = 0.0
    else:
        Lap[np.isnan(Lap)] = np.nanmean(Lap)
    lme = float(np.sum(Lap) - nparams*np.log(nsubjects))
    return Lap, lme, good

def calc_BICint(
    all_data, param_names, mu, sigma, fit_func, nsamples: int = 2000, func_output: str = "all", nll_key: str = "nll"
) -> float:
    npar = len(param_names)

    def subject_trials(beh) -> int:
        if isinstance(beh, pd.DataFrame):
            return len(beh)
        if isinstance(beh, np.ndarray):
            return int(beh.size)
        if isinstance(beh, (list, tuple)):
            for item in beh:
                if hasattr(item, "size"):
                    return int(item.size)
        raise ValueError("Unrecognized data structure in all_data")

    total_trials = int(np.sum([subject_trials(beh) for beh in all_data]))

    sigmasqrt = np.sqrt(np.asarray(sigma).reshape(-1))
    mu = np.asarray(mu).reshape(-1)

    def subj_iLog(beh):
        G = norm.rvs(loc=mu[:, None], scale=sigmasqrt[:, None], size=(len(mu), nsamples))
        subnll = []
        for k in range(nsamples):
            pars = G[:, k]
            # fit_func expected to return dict when output="all"
            info = fit_func(pars, *beh, output=func_output)
            subnll.append(info[nll_key])
        iLog = logsumexp(-np.asarray(subnll)) - np.log(nsamples)
        return iLog

    iLogs = Parallel(n_jobs=-1)(delayed(subj_iLog)(beh) for beh in all_data)
    iLogs = np.asarray(iLogs)
    finite = np.isfinite(iLogs)
    if not np.all(finite):
        iLogs = iLogs[finite]
    bicint = -2*np.sum(iLogs) + npar*np.log(total_trials)
    return float(bicint)

def pseudo_r2_from_nll(nll: np.ndarray, ntrials_total: int, noptions: int, metric: str = 'median') -> float:
    if metric not in {'median', 'mean'}:
        raise ValueError("metric must be 'median' or 'mean'")

    if metric == 'median':
        median_nll = float(np.median(nll))
        random_baseline = float(np.median(-np.log(1.0 / noptions) * ntrials_total))
        return 1.0 - (median_nll / random_baseline)
    else:
        mean_nll = float(np.mean(nll))
        random_baseline = float(np.mean(-np.log(1.0 / noptions) * ntrials_total))
        return 1.0 - (mean_nll / random_baseline)


def overall_predictive_probability_from_nll(
    nll: np.ndarray,
    nchoices_total: int,
    *,
    return_log: bool = False,
) -> float:
    """Compute geometric-mean predictive probability from summed NLL values.

    For per-subject summed negative log-likelihood values ``nll``, the overall
    predictive probability over all modeled choices is:

        p = exp(-sum(nll) / nchoices_total)

    This corresponds to the geometric mean of trial-level predictive
    probabilities across all subjects and trials.

    Parameters
    ----------
    nll : np.ndarray
        1D array containing per-subject summed negative log-likelihood values.
    nchoices_total : int
        Total number of modeled choices across all subjects.
    return_log : bool, default False
        If True, return ``log(p)`` instead of ``p``.

    Returns
    -------
    float
        Geometric mean predictive probability (or its log if ``return_log`` is
        True).
    """
    nll = np.asarray(nll, dtype=float)
    if nll.ndim != 1:
        raise ValueError("nll must be a 1D array of shape (nsubjects,)")
    if nchoices_total <= 0:
        raise ValueError("nchoices_total must be a positive integer")

    finite_nll = nll[np.isfinite(nll)]
    if finite_nll.size == 0:
        return float("nan")

    log_gmean = -float(np.sum(finite_nll)) / float(nchoices_total)
    if return_log:
        return log_gmean
    return float(np.exp(log_gmean))


def likelihood_r2(nll: np.ndarray, metric: str = 'median') -> float:
    """
    R^2-style score from per-subject summed negative log-likelihoods.

    Steps:
      1) Convert to per-subject joint likelihoods: L_i = exp(-nll_i)
      2) Aggregate across subjects using either the median (default) or mean
      3) Square the aggregate to get the final score

    Parameters
    ----------
    nll : np.ndarray
        1D array of shape (nsubjects,) with summed negative log-likelihoods.
        NaNs are ignored in aggregation.
    metric : {'median', 'mean'}, default 'median'
        Aggregation across subjects.

    Returns
    -------
    float
        R^2-like scalar (NaN if all inputs are NaN).
    """
    if metric not in {'median', 'mean'}:
        raise ValueError("metric must be 'median' or 'mean'")

    nll = np.asarray(nll, dtype=float)
    if nll.ndim != 1:
        raise ValueError("nll must be a 1D array of shape (nsubjects,)")

    with np.errstate(over='ignore', invalid='ignore'):
        likelihoods = np.exp(-nll)  # per-subject joint likelihood in (0, 1], NaN-preserving

    agg = np.nanmedian(likelihoods) if metric == 'median' else np.nanmean(likelihoods)
    return float(agg ** 2)
