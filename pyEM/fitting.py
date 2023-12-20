import numpy as np
from scipy.optimize import minimize
from copy import deepcopy
from scipy.stats import norm
from joblib import Parallel, delayed
from pyEM.math import compGauss_ms

def minimize_negLL(objfunc, behavioral_data, param_values, param_bounds):
    '''
    Vanilla MLE fit

    Inputs:
        - objfunc (function): function to be minimized, should output negative log likelihood
        - behavioral_data (list of lists): data to be fit, each item in list is a list containing numpy arrays for model fitting
        - param_values (list): parameter value guesses
        - param_bounds (list): bounds on parameters
    '''

    result = minimize(objfunc, 
                      param_values,
                      (x for x in behavioral_data),
                      bounds=(x for x in param_bounds))
    return result

def expectation_step(objfunc, objfunc_input, prior, nparams, **kwargs):
    '''
    Subject-wise model fit 

    Inputs:
        - objfunc (function): function to be minimized, should output negative log likelihood
        - objfunc_input (list): arguments for objfunc
        - prior (dict): prior mean and variance of parameters with logpdf function
        - nparams (int): number of parameters
    
    Returns:
        - q_est (np.array): estimated parameters
        - hess_mat (np.array): inverse hessian matrix
        - fval (float): negative posterior likelihood
        - nl_prior (float): negative log prior
    '''

    # check for kwargs
    for key, value in kwargs.items():
        # Check if the key is a valid variable name
        if not key.isidentifier():
            raise ValueError(f"'{key}' is not a valid variable name!")
        
        # Dynamically assign variable names (be cautious with this approach)
        locals()[key] = value 

    # fit model, calculate P(Choices | h) * P(h | O) 
    ex = -1
    tmp = 0

    while ex < 0:
        # Set free parameters to random values
        q = 0.1 * np.random.randn(nparams)

        # Perform the optimization
        result = minimize(objfunc, x0=q, 
                          args=tuple([x for x in objfunc_input]+[prior]))
        q_est = result.x
        fval  = result.fun
        ex    = result.status

        if ex < 0:
            tmp += 1
            print(f'didn\'t converge {tmp} times exit status {ex}')

    # Return fitted parameters and their hessians
    return q_est, result.hess_inv, fval, -prior['logpdf'](q_est)

def EMfit(all_data, objfunc, param_names, **kwargs):
    '''
    Expectation Maximization with MAP
    Adapted for Python from Marco Wittmann (2017), Patricia Lockwood & Miriam Klein-Flügge (2020), and Jo Cutler (2021)

    Inputs:
        - all_data (list of lists): data to be fit, each item in list is a list containing numpy arrays for model fitting
        - objfunc (dict): function to be minimized, should output negative log likelihood; this carries to expectation_step
        - param_names (list): parameters names as strings; e.g.: ['beta', 'lr']
    
    Returns:
        - m (np.array): estimated parameters
        - inv_h (np.array): inverse hessian matrix
        - posterior (dict): posterior mean and variance of parameters with logpdf function
        - NPL (np.array): negative posterior likelihood
        - NLPrior (np.array): negative log prior
        - NLL (np.array): negative log likelihood
    '''
    # Set the number of fitting iterations
    max_iterations = 800
    convCrit       = .001
    nparams        = len(param_names)
    nsubjects      = len(all_data)

    # Initialize group-level parameter mean and variance
    posterior = {}
    posterior['mu']    = np.abs(0.1 * np.random.randn(nparams, 1))
    posterior['sigma'] = np.full((nparams, 1), 100)

    # get kwargs if any
    for key, value in kwargs.items():
        # Check if the key is a valid variable name
        if not key.isidentifier():
            raise ValueError(f"'{key}' is not a valid variable name!")
        
        # Dynamically assign variable names (be cautious with this approach)
        locals()[key] = value

    # initialise transient variables:
    nextbreak  = 0
    NPL_old    = -np.inf
    prior      = {}
    NPL        = np.zeros((nsubjects,1)) #posterior ikelihood
    NLL        = np.zeros((nsubjects,1)) #NLL estimate per iteration
    NLPrior    = np.zeros((nsubjects,1)) #negative LogPrior estimate per iteration

    NPL_list = []
    for iiter in range(max_iterations):

        # individual-level parameter mean estimate
        m = np.zeros((nparams,nsubjects))

        # individual-level parameter variance estimate
        inv_h = np.zeros((nparams, nparams,nsubjects))

        # Assume you have the 'posterior' dictionary with 'mu' and 'sigma' keys
        # Build prior gaussian pdfs to calculate P(h|O):
        prior['mu'] = deepcopy(posterior['mu'])
        prior['sigma'] = deepcopy(posterior['sigma'].copy())
        prior['logpdf'] = lambda x: np.sum(norm.logpdf(x, prior['mu'], 
                                                        np.sqrt(prior['sigma']))) #calculates separate normpdfs per parameter and sums their logs
        
        # ------ EXPECTATION STEP -------------------------------------------------
        # Loop over subjects
        results = Parallel(n_jobs=-1)(delayed(expectation_step)(objfunc, all_data[subj_idx], prior, nparams) for subj_idx in range(nsubjects))

        # Store the results
        this_NPL = np.zeros((nsubjects,1))
        this_NLPrior = np.zeros((nsubjects,1))
        for subj_idx, (q_est, hess_mat, fval, nl_prior) in enumerate(results):
            m[:,subj_idx]     = deepcopy(q_est)
            inv_h[:, :,subj_idx]  = deepcopy(hess_mat)
            this_NPL[subj_idx]      = deepcopy(fval)
            this_NLPrior[subj_idx]  = deepcopy(nl_prior)
        
        if iiter == 0:
            NPL = deepcopy(this_NPL)
            NLPrior = deepcopy(this_NLPrior)
        else:
            NPL = np.hstack((NPL, this_NPL))
            NLPrior = np.hstack((NLPrior, this_NLPrior))

        # ------ MAXIMIZATION STEP -------------------------------------------------
        # compute gaussians and sigmas per parameter
        curmu, cursigma, flagcov, _ = compGauss_ms(m,inv_h)

        if flagcov == 1:
            posterior['mu'] = deepcopy(curmu)
            posterior['sigma'] = deepcopy(cursigma)

        if sum(NPL[:,iiter]) == 10000000 * nsubjects:
            flagcov = 0
            print('-[badfit]-')

        # check whether fit has converged
        NPL_list += [sum(NPL[:,iiter])]
        if sum(NPL[:,iiter]) <= min(NPL_list):
            print(f'{sum(NPL[:,iiter]):.3f} ({iiter:03d})', end=', ')
        
        if abs(sum(NPL[:,iiter]) - NPL_old) < convCrit and flagcov == 1:
            print(' -- converged!!!!! ')
            nextbreak = 1

        NPL_old = sum(NPL[:,iiter])

        if nextbreak == 1:
            break 

        if iiter == (max_iterations-1):
            print('-maximum number of iterations reached\n')

    return m, inv_h, posterior, NPL, NLPrior, NLL