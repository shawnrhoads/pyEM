
from __future__ import annotations
import numpy as np
from scipy.stats import norm
from ..utils.math import norm2alpha, calc_fval
from scipy.special import expit

def glm_sim(params: np.ndarray, ntrials: int = 100):
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

def glm_fit(params, X, Y, prior=None, output: str = 'npl'):
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

def glm_decay_sim(params, ntrials: int = 100):
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

def glm_decay_fit(params, X, Y, prior=None, output: str = 'npl', decay: str = 'twostep'):
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

def logit_sim(params: np.ndarray, ntrials: int = 100):
    """Simulate data for a standard logistic regression.

    Args:
        params: array (n_obs, nparams). First column should be intercept weight.
        ntrials: number of trials per observation.

    Returns:
        X: (n_obs, ntrials, nparams)
        Y: (n_obs, ntrials) of 0/1 draws
    """
    n_obs, nparams = params.shape
    Y = np.zeros((n_obs, ntrials), dtype=int)
    X = np.zeros((n_obs, ntrials, nparams))
    rng = np.random.default_rng(2021)
    for s in range(n_obs):
        X[s, :, :] = np.concatenate(
            [np.ones((ntrials, 1)), rng.normal(size=(ntrials, nparams - 1))],
            axis=1,
        )
        p = expit(X[s].dot(params[s]))
        Y[s, :] = rng.binomial(1, p)
    return X, Y

def logit_fit(params, X, Y, prior=None, output: str = 'npl'):
    """Negative log-likelihood for logistic regression (no decay).

    Args:
        params: parameter vector (nparams,)
        X: (ntrials, nparams)
        Y: (ntrials,) of 0/1
        prior: optional prior dict passed to calc_fval
        output: 'npl'/'nll' for scalar objective, 'all' for details
    """
    logits = X.dot(params)
    p = expit(logits)
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    negll = -np.sum(Y * np.log(p) + (1 - Y) * np.log(1 - p))
    if output in ('npl', 'nll'):
        return calc_fval(negll, params, prior=prior, output=output)
    elif output == 'all':
        return {
            'params': params,
            'predicted_p': p,
            'negll': negll,
            'BIC': len(params) * np.log(len(Y)) + 2 * negll,
        }

def logit_decay_sim(params: np.ndarray, ntrials: int = 100):
    """Simulate logistic-regression data with exponentially discounted predictors.

    Args:
        params: array (n_obs, nparams_with_gamma). Last column is gamma in [0,1].
                The remaining columns are regression weights (including intercept).
        ntrials: number of trials per observation.

    Returns:
        X: (n_obs, ntrials, nparams) base predictors (not discounted)
        Y: (n_obs, ntrials) of 0/1 draws
    """
    n_obs, nparams_with_gamma = params.shape
    nparams = nparams_with_gamma - 1
    Y = np.zeros((n_obs, ntrials), dtype=int)
    X = np.zeros((n_obs, ntrials, nparams))
    rng = np.random.default_rng(2021)
    for s in range(n_obs):
        gamma = params[s, -1]            # assume already in [0,1] for simulation
        pv = params[s, :-1]              # includes intercept weight
        X[s, :, :] = np.concatenate(
            [np.ones((ntrials, 1)), rng.normal(size=(ntrials, nparams - 1))],
            axis=1,
        )
        for t in range(ntrials):
            discounted = np.zeros(nparams)
            for j in range(3):
                if t - j >= 0:
                    discounted += (gamma ** j) * X[s, t - j, :]
            p = expit(discounted.dot(pv))
            Y[s, t] = rng.binomial(1, p)
    return X, Y

def logit_decay_fit(
    params,
    X,
    Y,
    prior=None,
    output: str = 'npl',
    decay: str = 'twostep'
):
    """Logistic regression with exponentially decaying regressors.

    ``params`` contains the regression weights followed by a discount factor
    (in Gaussian space). The discount factor is mapped to (0,1) via norm2alpha.
    ``decay`` determines whether two or three previous trials influence the
    current prediction (matches your GLM template).
      - 'twostep' -> use j = 0,1,2 (3 terms)
      - otherwise -> use j = 0,1   (2 terms)

    Args:
        params: vector of length (nreg + 1). Last element is gamma in Gaussian space.
        X: (ntrials, nreg) base predictors at each trial (not discounted).
        Y: (ntrials,) of 0/1.
        prior: optional prior dict passed to calc_fval.
        output: 'npl'/'nll' for scalar objective, 'all' for details.
        decay: 'twostep' or anything else (mirrors glm_decay_fit behavior).
    """
    decay_j = 3 if decay == 'twostep' else 2
    ntrials, nreg = X.shape
    nparams = len(params) - 1
    pv = params[:-1]
    gamma = norm2alpha(params[-1])  # map R -> (0,1)

    # guard against numerical issues
    if gamma < 0 or gamma > 1:
        return 1e7

    logits = np.zeros(ntrials)
    for t in range(ntrials):
        discounted = np.zeros(nreg)
        for j in range(decay_j):
            if t - j >= 0:
                discounted += (gamma ** j) * X[t - j, :]
        logits[t] = discounted.dot(pv)

    p = expit(logits)
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    negll = -np.sum(Y * np.log(p) + (1 - Y) * np.log(1 - p))

    if output in ('npl', 'nll'):
        return calc_fval(negll, params, prior=prior, output=output)
    elif output == 'all':
        return {
            'params': params,
            'predicted_p': p,
            'negll': negll,
            'BIC': len(params) * np.log(len(Y)) + 2 * negll,
            'gamma': gamma,
        }

def glm_ar_sim(params: np.ndarray, ntrials: int = 100):
    """
    Generate data from a linear regression model with an AR(1) term.
    params: shape (n_obs, nparams). Last column is AR(1) coefficient phi.
    """
    n_obs, nparams = params.shape
    phi_idx = nparams - 1  # last parameter = AR coefficient
    beta_idx = np.arange(nparams - 1)  # all but last param

    Y = np.zeros((n_obs, ntrials))
    X = np.zeros((n_obs, ntrials, nparams - 1))
    rng = np.random.default_rng(2021)

    for s in range(n_obs):
        # predictors: intercept + random covariates (exclude AR param)
        X[s, :, :] = np.concatenate(
            [np.ones((ntrials, 1)), rng.normal(size=(ntrials, nparams - 2))],
            axis=1
        )

        beta = params[s, beta_idx]
        phi = params[s, phi_idx]

        # baseline linear part
        lin = X[s].dot(beta)

        # apply AR(1) recursion: y_t = lin_t + phi * y_{t-1} + noise
        y = np.zeros(ntrials)
        y[0] = lin[0] + rng.normal()  # initial
        for t in range(1, ntrials):
            y[t] = lin[t] + phi * y[t-1] + rng.normal()

        Y[s, :] = y

    return X, Y

def glm_ar_fit(params, X, Y, prior=None, output: str = 'npl'):
    """
    Negative log-likelihood for Gaussian regression with an AR(1) term.
    params: includes intercept, betas..., phi
    """
    beta = params[:-1]
    phi = params[-1]

    # linear predictor without AR part
    lin = X.dot(beta)

    # construct AR(1) predictions over time
    pred = np.zeros_like(Y)
    pred[0] = lin[0]
    for t in range(1, len(Y)):
        pred[t] = lin[t] + phi * Y[t-1]  # AR uses observed y_{t-1}

    resid_sigma = np.std(Y - pred)
    negll = -np.sum(norm.logpdf(Y, loc=pred, scale=resid_sigma))

    if output in ('npl', 'nll'):
        return calc_fval(negll, params, prior=prior, output=output)
    elif output == 'all':
        return {
            'params': np.hstack([beta, phi]),
            'predicted_y': pred,
            'negll': negll,
            'BIC': len(params) * np.log(len(Y)) + 2 * negll,
        }
