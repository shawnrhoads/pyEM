import numpy as np
from scipy.special import expit

def softmax(EVs, beta):
    if type(EVs) is list:
        EVs = np.array(EVs)
    return np.exp(beta*EVs) / np.sum(np.exp(beta*EVs))

def norm2beta(beta):
    # return maxval / (1 + np.exp(-beta))
    return np.exp(beta)

def beta2norm(beta):
    # return np.log(beta / (maxval - beta))
    return np.log(beta)

def norm2alpha(alpha_norm):
    return expit(alpha_norm)

def alpha2norm(alpha):
    return -np.log(1.0/alpha - 1.0)

def compGauss_ms(m, h, vargin=None):
    # compute group-level gaussian from fminunc computed parameters and their covariances
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
