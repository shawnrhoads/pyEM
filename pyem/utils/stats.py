
from __future__ import annotations
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from scipy.special import logsumexp

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
    all_data, param_names, mu, sigma, fit_func, nsamples: int = 2000, func_output: str = "all",
    nll_key: str = "nll", ntrials_total: int | None = None,
) -> float:
    """Integrated BIC via Monte Carlo integration over the group posterior.

    ``ntrials_total`` is the number of independent trials contributing to a
    single subject's likelihood, used for the ``k*log(n)`` complexity
    penalty. If not supplied, it is auto-detected from the first subject's
    data, assuming every array-like field in that subject's data is
    trial-aligned with identical shape (true for the RW/Bayes families,
    e.g. ``[choices, rewards]`` both shaped ``(nblocks, ntrials)``). For
    families with heterogeneous per-trial shapes (e.g. a GLM's ``[X, Y]``
    where ``X`` has an extra feature-count dimension), pass
    ``ntrials_total`` explicitly rather than relying on auto-detection.
    """
    npar = len(param_names)
    if ntrials_total is not None:
        total_trials = ntrials_total
    elif isinstance(all_data[0], pd.DataFrame):
        total_trials = len(all_data[0])
    else:
        first = all_data[0]
        if isinstance(first, (list, tuple)) and hasattr(first[0], "size"):
            total_trials = int(first[0].size)
        else:
            raise ValueError("Unrecognized data structure in all_data")
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
        # log(sum(exp(-subnll))/nsamples), computed via logsumexp for numerical
        # stability: the naive form over/underflows when subnll (an unbounded
        # NLL, e.g. from a Gaussian-likelihood GLM fit) is very negative or
        # very positive, silently turning a valid subject into inf/-inf that
        # then gets dropped below instead of contributing its real value.
        iLog = logsumexp(-np.asarray(subnll)) - np.log(nsamples)
        return iLog
    iLogs = Parallel(n_jobs=-1)(delayed(subj_iLog)(beh) for beh in all_data)
    iLogs = np.asarray(iLogs)
    finite = np.isfinite(iLogs)
    if not np.all(finite):
        print(f"calc_BICint: dropping {int((~finite).sum())}/{len(iLogs)} subject(s) with non-finite integrated log-likelihood")
        iLogs = iLogs[finite]
    bicint = -2*np.sum(iLogs) + npar*np.log(total_trials)
    return float(bicint)

def pseudo_r2_from_nll(nll: np.ndarray, ntrials_total: int, noptions: int, metric: str = 'median') -> float:
    if metric == 'median':
        median_nll = float(np.median(nll))
        random_baseline = float(np.median(-np.log(1.0 / noptions) * ntrials_total))
        return 1.0 - (median_nll / random_baseline)
    else:
        mean_nll = float(np.mean(nll))
        random_baseline = float(np.mean(-np.log(1.0 / noptions) * ntrials_total))
        return 1.0 - (mean_nll / random_baseline)

