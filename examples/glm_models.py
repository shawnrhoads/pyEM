import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import sys
sys.path.append('../')
from pyEM.math import norm2alpha

def simulate(params, ntrials=100):
    '''
    Simulates a general linear model (e.g., Y ~ b0 + b1*X1 + b2*X2 + ...) with params.shape[1] features
    '''
    n_observations, nparams = params.shape

    Y_out = np.zeros((n_observations, ntrials))
    X_out = np.zeros((n_observations, ntrials, nparams))
    for subj in tqdm(range(n_observations)):
        # Generate data
        np.random.seed(2021)

        # Random X with intercept
        X_out[subj,:,:] = np.concatenate((np.ones((ntrials, 1)), 
                    np.random.normal(size=(ntrials, nparams-1))), axis=1)

        # Y as function of X + noise
        Y_out[subj,:] = np.dot(X_out[subj,:], params[subj,:]) + np.random.normal(size=(ntrials,))

    return X_out, Y_out

def fit(params, X, Y, prior=None, output='npl'):
    '''
    Estimates coefficients of simple GLM (e.g., Y ~ b0 + b1*X1 + b2*X2 + ...)
    '''

    # Fit GLM
    predicted_y = np.dot(X, params) 
    
    # Compute neg log likelihood with stats.norm.logpdf
    resid_sigma = np.std(np.subtract(Y, predicted_y)) # std dev of residuals
    negll = -np.sum(norm.logpdf(Y, loc=predicted_y, scale=resid_sigma))

    if output == 'npl':
        if prior is not None:  # EM-fit: P(Choices | h) * P(h | O) should be maximised, therefore same as minimizing it with negative sign
            fval = -(-negll + prior['logpdf'](params))

            if np.isinf(fval):
                fval = 10000000

            if fval is None:
                fval = 10000000

            return fval
        else: # NLL fit 
            return negll
    elif output == 'all':
        subj_dict = {'params'     : params,
                     'predicted_y': predicted_y,
                     'negll'      : negll,
                     'BIC'        : len(params) * np.log(len(Y)) + 2*negll}
        return subj_dict
    
def simulate_decay(params, ntrials=100):
    '''
    Simulates a general linear model with discounting of previous values of Xn by gamma.
    Y ~ b0 + gamma^0 * b1*X1_t + gamma^1 * b1*X1_{t-1} + gamma^2 * b1*X1_{t-2} + ...
    
    `gamma` is included as the last index in `params`.
    '''
    n_observations, nparams_with_gamma = params.shape
    nparams = nparams_with_gamma - 1  # The last parameter is gamma

    Y_out = np.zeros((n_observations, ntrials))
    X_out = np.zeros((n_observations, ntrials, nparams))
    
    for subj in range(n_observations):
        # Extract gamma for the current subject
        gamma = params[subj, -1]
        param_values = params[subj, :-1]  # Exclude gamma
        
        # Generate random data with an intercept
        np.random.seed(2021)
        X_out[subj,:,:] = np.concatenate((np.ones((ntrials, 1)), 
                                          np.random.normal(size=(ntrials, nparams-1))), axis=1)

        # Loop over trials and apply gamma discounting for past values of X
        for t in range(ntrials):
            discounted_sum = np.zeros(nparams)
            for j in range(3):  # Looking back 2 trials, i.e., j=0 (current), j=1 (previous), j=2 (two steps back)
                if t - j >= 0:  # Ensure we're not indexing before the first trial
                    discounted_sum += (gamma ** j) * X_out[subj, t-j, :]
                    
            # Compute Y as a function of the discounted X values
            Y_out[subj, t] = np.dot(discounted_sum, param_values) + np.random.normal()
    
    return X_out, Y_out

def fit_decay(params, X, Y, prior=None, output='npl'):
    '''
    Estimates coefficients of a GLM with discounted Xn terms using gamma.
    Y ~ b0 + gamma^0 * b1*X1_t + gamma^1 * b1*X1_{t-1} + gamma^2 * b1*X1_{t-2} + ...
    
    `gamma` is included as the last index in `params`.
    '''
    
    ntrials, n_regressors = X.shape
    nparams_with_gamma = len(params)
    nparams = nparams_with_gamma - 1  # The last parameter is gamma

    predicted_y = np.zeros((ntrials,))
    
    # Extract gamma and the parameter values for the current subject
    gamma = norm2alpha(params[-1]) # transform to be between 0 and 1
    if gamma < 0 or gamma > 1:
        return 10000000

    param_values = params[:-1]  # Exclude gamma
    
    for t in range(ntrials):
        discounted_sum = np.zeros(nparams)
        for j in range(3):  # Discount over current and past two trials
            if t - j >= 0:
                discounted_sum += (gamma ** j) * X[t-j, :]

        # Compute predicted Y as the dot product of the discounted X terms
        predicted_y[t] = np.dot(discounted_sum, param_values)
    
    # Compute negative log likelihood using stats.norm.logpdf
    resid_sigma = np.std(np.subtract(Y, predicted_y))  # Standard deviation of residuals
    negll = -np.sum(norm.logpdf(Y, loc=predicted_y, scale=resid_sigma))

    if output == 'npl':
        if prior is not None:  # Include prior if specified
            fval = -(-negll + prior['logpdf'](params))
            return fval if not np.isinf(fval) and fval is not None else 10000000
        else:
            return negll
    elif output == 'all':
        return {
            'params': params,
            'predicted_y': predicted_y,
            'negll': negll,
            'BIC': len(params) * np.log(len(Y)) + 2 * negll
        }
