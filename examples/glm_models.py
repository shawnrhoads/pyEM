import numpy as np
from tqdm import tqdm
from scipy.stats import norm

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