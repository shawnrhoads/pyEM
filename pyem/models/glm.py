
from __future__ import annotations
import numpy as np
from scipy.stats import norm
from ..utils.math import norm2alpha
from ..utils.stats import calc_BICint  # exposed for convenience

def simulate(params: np.ndarray, ntrials: int = 100):
    n_obs, nparams = params.shape
    Y = np.zeros((n_obs, ntrials))
    X = np.zeros((n_obs, ntrials, nparams))
    rng = np.random.default_rng(2021)
    for s in range(n_obs):
        X[s, :, :] = np.concatenate([np.ones((ntrials, 1)), rng.normal(size=(ntrials, nparams-1))], axis=1)
        Y[s, :] = X[s].dot(params[s]) + rng.normal(size=(ntrials,))
    return X, Y

def fit(params, X, Y, prior=None, output='npl'):
    pred = X.dot(params)
    resid_sigma = np.std(Y - pred)
    negll = -np.sum(norm.logpdf(Y, loc=pred, scale=resid_sigma))
    if output in ('npl','nll'):
        if prior is not None and output == 'npl':
            # simple GaussianPrior-like dict compat
            fval = negll + (-prior.logpdf(np.asarray(params))) if hasattr(prior, "logpdf") else negll
        else:
            fval = negll
        return fval
    elif output == 'all':
        return {'params': params, 'predicted_y': pred, 'negll': negll, 'BIC': len(params)*np.log(len(Y)) + 2*negll}

def simulate_decay(params, ntrials=100):
    n_obs, nparams_with_gamma = params.shape
    nparams = nparams_with_gamma - 1
    Y = np.zeros((n_obs, ntrials))
    X = np.zeros((n_obs, ntrials, nparams))
    rng = np.random.default_rng(2021)
    for s in range(n_obs):
        gamma = params[s, -1]
        pv = params[s, :-1]
        X[s, :, :] = np.concatenate([np.ones((ntrials, 1)), rng.normal(size=(ntrials, nparams-1))], axis=1)
        for t in range(ntrials):
            discounted = np.zeros(nparams)
            for j in range(3):
                if t - j >= 0:
                    discounted += (gamma**j) * X[s, t-j, :]
            Y[s, t] = discounted.dot(pv) + rng.normal()
    return X, Y

def fit_decay(params, X, Y, prior=None, output='npl', decay='twostep'):
    decay_j = 3 if decay=='twostep' else 2
    ntrials, nreg = X.shape
    nparams = len(params) - 1
    predicted_y = np.zeros(ntrials)
    gamma = norm2alpha(params[-1])
    if gamma < 0 or gamma > 1:
        return 1e7
    pv = params[:-1]
    for t in range(ntrials):
        discounted = np.zeros(nparams)
        for j in range(decay_j):
            if t-j >= 0:
                discounted += (gamma**j) * X[t-j, :]
        predicted_y[t] = discounted.dot(pv)
    resid_sigma = np.std(Y - predicted_y)
    negll = -np.sum(norm.logpdf(Y, loc=predicted_y, scale=resid_sigma))
    if output in ('npl','nll'):
        if prior is not None and output == 'npl' and hasattr(prior, 'logpdf'):
            return negll + (-prior.logpdf(np.asarray(params)))
        return negll
    elif output == 'all':
        return {'params': params, 'predicted_y': predicted_y, 'nll': negll, 'BIC': len(params)*np.log(len(Y)) + 2*negll}
