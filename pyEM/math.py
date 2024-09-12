import numpy as np
from scipy.special import expit
from scipy.stats import norm
from joblib import Parallel, delayed

def softmax(EVs, beta):
    if type(EVs) is list:
        EVs = np.array(EVs)
    return np.exp(beta*EVs) / np.sum(np.exp(beta*EVs))

def norm2beta(beta):
    return 4 / (1 + np.exp(-beta))

def beta2norm(beta):
    return np.log(beta / (10 - beta))
    # return np.log(beta)

def norm2alpha(alpha_norm):
    return expit(alpha_norm)

def alpha2norm(alpha):
    return -np.log(1.0/alpha - 1.0)

def calc_fval(negll, prior, params, output='npl'):
    if output == 'npl':
        if prior is not None:
            # P(Choices | h) * P(h | O) should be maximized, therefore same as minimizing it with negative sign
            fval = -(-negll + prior['logpdf'](params))            
            if np.isinf(fval):
                fval = 10000000
            return fval
        else: # NLL fit 
            return negll
        
def compGauss_ms(m, h, vargin=None):
    '''
    Computes group-level gaussian from computed parameters and their covariances
    Edited from Marco Wittmann (2017)

    Inputs:
        - m (np.array):  fitted parameters (npar x nsub matrix)
        - h (np.array):  individual-level inverse hessians (npar x npar x nsub)
        - vargin (int): if set to 2, computes covariance matrix in addition
    
    Outputs:
        - mu (np.array): group mu
        - sigma (np.array): group sigma
        - flagsigma (int): flag indicating whether model variance was calculated successfully
        - covmat (np.array): full covariance matrix; is [] if no vargin specified
    '''
    # 
    # MKW, 2017
    #
    # INPUT:    - m:  fitted parameters (npar x nsub matrix)
    #           - h:  individual-level inverse hessians (npar x npar x nsub)
    #           - vargin: if set to 2, computes covariance matrix in addition
    #
    # OUTPUT:   - group mu and group sigma
    #           - flagcov: flag indicating whether model variance was calculated successfully
    #           - covmat: full covariance matrix; is [] if no vargin specified
    # Ensure that m and h have the same number of subjects
    assert m.shape[1] == h.shape[2], "Mismatch in the number of subjects between m and h."

    # Ensure that m and h have the same number of parameters
    assert m.shape[0] == h.shape[0] == h.shape[1], "Mismatch in the number of parameters between m and h."

    # get info
    nsub = m.shape[1]
    npar = m.shape[0]
    covmat = None

    # ------ 1) compute mean: -------------------------------------------------
    mu = np.mean(m, axis=1)

    # ------2) Compute sigma: -------------------------------------------------
    sigma = np.zeros(npar) #sigma   = zeros(size(h,1),1);

    # compute sigma for each parameter
    for isub in range(nsub):
        sigma += m[:, isub] ** 2 + np.diag(h[:, :, isub])
    sigma = (sigma/nsub) - mu ** 2

    # give error message in case:
    flagsigma = 1
    if np.min(sigma) < 0:
        flagsigma = 0
        print('..CovError!')

    # ----- 3) Optional: Get full covariance matrix----------------------------
    if vargin is None:
        return mu, sigma, flagsigma, covmat

    covmat = np.zeros((npar, npar))
    if vargin == 2:
        for isub in range(nsub):
            covmat += np.outer(m[:, isub], m[:, isub]) - np.outer(m[:, isub], mu) - np.outer(mu, m[:, isub]) + np.outer(mu, mu) + h[:, :, isub]

        covmat /= nsub

    if np.linalg.det(covmat) <= 0:
        print('Negative/zero determinant - prior covariance not updated')

    return mu, sigma, flagsigma, covmat

def calc_BICint(all_data, param_names, mu, sigma, fit_func, nsamples=2000):
    """
    Calculates the integrated BIC.

    Parameters:
        all_data (list): A list of lists of behavior data arrays with shape (nblocks, ntrials) for each subject e.g., [[choices, rewards], [choices, rewards]].
        param_names (list): List of parameter names.
        mu (numpy.ndarray): Array of parameter mean estimates of sample with shape (n_params,) from posterior.
        sigma (numpy.ndarray): Array of parameter variances of sample with shape (n_params, ) from posterior.
        fit_func (callable): A function that fits the model given a sample of parameters and outputs a dictionary containing the key 'negll' corresponding to the negative log-likelihood (NLL).
        nsamples (int, optional): Number of samples drawn. Defaults to 2000.

    Returns:
        bicint (float): Integrated BIC value for model.

    Example: `bicint = calc_BICint(all_data, param_names, posterior['mu'], posterior['sigma'], rw_models.fit)`
    
    """
    # Define settings
    npar = len(param_names)
    nblocks, ntrials = all_data[0][0].shape

    # Convert to std dev
    sigmasqrt = np.sqrt(sigma)

    # Initialize
    iLog = np.empty(len(all_data))
    
    # Start computing
    for isub, beh in enumerate(all_data):        
        # Sample parameters from the Gaussian distribution
        Gsamples = norm.rvs(loc=np.tile(mu[:, np.newaxis], (1, nsamples)), scale=np.tile(sigmasqrt[:, np.newaxis], (1, nsamples)))

        # Compute negative log likelihood for each sample
        subnll = Parallel(n_jobs=-1)(delayed(lambda k: fit_func(*([Gsamples[:, k]] + beh), output='all')['negll'])(k) for k in range(nsamples))

        # Compute integrated log likelihood
        iLog[isub] = np.log(np.sum(np.exp(-np.array(subnll))) / nsamples)

    # Compute BICint
    bicint = -2 * np.sum(iLog) + npar * np.log(ntrials*nblocks)

    return bicint