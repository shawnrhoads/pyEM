
from __future__ import annotations
import numpy as np
from scipy.stats import norm
from ..utils.math import norm2alpha, calc_fval

def simulate(params: np.ndarray, ntrials: int = 100):
    """Generate data from a standard linear regression model."""
    n_obs, nparams = params.shape
    Y = np.zeros((n_obs, ntrials))
    X = np.zeros((n_obs, ntrials, nparams))
    rng = np.random.default_rng(2021)
    for s in range(n_obs):
        # First column is intercept; remaining columns are random predictors
        X[s, :, :] = np.concatenate(
            [np.ones((ntrials, 1)), rng.normal(size=(ntrials, nparams - 1))], axis=1
        )
        Y[s, :] = X[s].dot(params[s]) + rng.normal(size=(ntrials,))
    return X, Y

def fit(params, X, Y, prior=None, output: str = 'npl'):
    """Negative log-likelihood for a Gaussian GLM."""
    pred = X.dot(params)
    resid_sigma = np.std(Y - pred)
    negll = -np.sum(norm.logpdf(Y, loc=pred, scale=resid_sigma))
    if output in ('npl', 'nll'):
        return calc_fval(negll, params, prior=prior, output=output)
    elif output == 'all':
        return {
            'params': params,
            'predicted_y': pred,
            'negll': negll,
            'BIC': len(params) * np.log(len(Y)) + 2 * negll,
        }

def simulate_decay(params, ntrials: int = 100):
    """Simulate GLM data with exponentially discounted predictors."""
    n_obs, nparams_with_gamma = params.shape
    nparams = nparams_with_gamma - 1
    Y = np.zeros((n_obs, ntrials))
    X = np.zeros((n_obs, ntrials, nparams))
    rng = np.random.default_rng(2021)
    for s in range(n_obs):
        gamma = params[s, -1]
        pv = params[s, :-1]
        X[s, :, :] = np.concatenate(
            [np.ones((ntrials, 1)), rng.normal(size=(ntrials, nparams - 1))], axis=1
        )
        for t in range(ntrials):
            discounted = np.zeros(nparams)
            for j in range(3):
                if t - j >= 0:
                    discounted += (gamma ** j) * X[s, t - j, :]
            Y[s, t] = discounted.dot(pv) + rng.normal()
    return X, Y

def fit_decay(params, X, Y, prior=None, output: str = 'npl', decay: str = 'twostep'):
    """GLM with exponentially decaying regressors.

    ``params`` contains the regression weights followed by a discount factor
    (in Gaussian space).  ``decay`` determines whether two or three previous
    trials influence the current prediction.
    """
    decay_j = 3 if decay == 'twostep' else 2
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
            if t - j >= 0:
                discounted += (gamma ** j) * X[t - j, :]
        predicted_y[t] = discounted.dot(pv)
    resid_sigma = np.std(Y - predicted_y)
    negll = -np.sum(norm.logpdf(Y, loc=predicted_y, scale=resid_sigma))
    if output in ('npl', 'nll'):
        return calc_fval(negll, params, prior=prior, output=output)
    elif output == 'all':
        return {
            'params': params,
            'predicted_y': predicted_y,
            'nll': negll,
            'BIC': len(params) * np.log(len(Y)) + 2 * negll,
        }

