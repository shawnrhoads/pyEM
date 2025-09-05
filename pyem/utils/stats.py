
from __future__ import annotations
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
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
    all_data, param_names, mu, sigma, fit_func, nsamples: int = 500, func_output: str = "all", nll_key: str = "NLL"
) -> float:
    npar = len(param_names)
    # count trials of the first subject
    if isinstance(all_data[0], pd.DataFrame):
        total_trials = len(all_data[0])
    else:
        first = all_data[0]
        if isinstance(first, (list, tuple)) and hasattr(first[0], "size"):
            total_trials = int(np.sum([x.size for x in first if hasattr(x, "size")]))
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
        iLog = np.log(np.sum(np.exp(-np.asarray(subnll))) / nsamples)
        return iLog
    iLogs = Parallel(n_jobs=-1)(delayed(subj_iLog)(beh) for beh in all_data)
    iLogs = np.asarray(iLogs)
    finite = np.isfinite(iLogs)
    if not np.all(finite):
        iLogs = iLogs[finite]
    bicint = -2*np.sum(iLogs) + npar*np.log(total_trials)
    return float(bicint)

def pseudo_r2_from_nll(nll: np.ndarray, ntrials: int, nopts: int) -> float:
    L = float(np.mean(nll))
    R = -np.log(1.0 / nopts) * ntrials
    return 1.0 - (L / R)
