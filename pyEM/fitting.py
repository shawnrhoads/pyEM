import numpy as np
from scipy.optimize import minimize
from copy import deepcopy
from scipy.stats import norm
from joblib import Parallel, delayed
from pyEM.math import compGauss_ms

def minimize_negLL(objfunc, param_values, param_bounds, behavioral_data, **kwargs):
    # vanilla MLE fit
    # objective_func (function): function to be minimized, should output negative log likelihood
    # param_values (list): initial parameter values
    # param_bounds (list): bounds on parameters
    # behavioral_data (list of lists): data to be fit

    result = minimize(objfunc, 
                      param_values,
                      (x for x in behavioral_data),
                      bounds=(x for x in param_bounds))
    return result

def expectation_step(objfunc, objfunc_input, prior, nparams, param_bounds, **kwargs):
    # subject-wise model fit 
    # objective_func (function): function to be minimized, should output negative log likelihood
    # func_args (dict): arguments for objective_func

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
    # return [x for x in objfunc_input]+[prior]
        result = minimize(objfunc, x0=q, 
                          args=tuple([x for x in objfunc_input]+[prior]), 
                          bounds=param_bounds)
        q_est = result.x
        fval  = result.fun
        ex    = result.status

        if ex < 0:
            tmp += 1
            print(f'didn\'t converge {tmp} times exit status {ex}')

    # Return fitted parameters and their hessians
    return q_est, result.hess_inv, fval, -prior['logpdf'](q_est)

def EMfit(all_data, objfunc, param_bounds, **kwargs):
    '''
    all_data (dict): data to be fit, each key is a subject ID
    objfunc (function): function to be minimized, should output negative log likelihood
    param_bounds (list): bounds on parameters
    '''
    
    # Set the number of fitting iterations
    max_iterations = 800
    convCrit       = .001
    nparams        = len(param_bounds)
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
    nextbreak   = 0
    NPL_old     = -np.inf
    prior       = {}
    NPL      = np.zeros((nsubjects,1)) #posterior ikelihood
    NLL      = np.zeros((nsubjects,1)) #NLL estimate per iteration
    NLPrior  = np.zeros((nsubjects,1)) #negative LogPrior estimate per iteration

    NPL_list = []
    for iiter in range(max_iterations):

        # individual-level parameter mean estimate
        m = np.zeros((nparams,nsubjects))

        # individual-level parameter variance estimate
        h = np.zeros((nparams, nparams,nsubjects))

        # Assume you have the 'posterior' dictionary with 'mu' and 'sigma' keys
        # Build prior gaussian pdfs to calculate P(h|O):
        prior['mu'] = deepcopy(posterior['mu'])
        prior['sigma'] = deepcopy(posterior['sigma'].copy())
        prior['logpdf'] = lambda x: np.sum(norm.logpdf(x, prior['mu'], 
                                                        np.sqrt(prior['sigma']))) #calculates separate normpdfs per parameter and sums their logs

        ### EXPECTATION STEP
        # Loop over subjects
        results = Parallel(n_jobs=-1)(delayed(expectation_step)(objfunc, all_data[subj_idx], prior, nparams, param_bounds) for subj_idx in range(nsubjects))

        # Store the results
        this_NPL = np.zeros((nsubjects,1))
        this_NLPrior = np.zeros((nsubjects,1))
        for subj_idx, (q_est, hess_mat, fval, nl_prior) in enumerate(results):
            m[:,subj_idx]     = deepcopy(q_est)
            h[:, :,subj_idx]  = deepcopy(hess_mat.todense())
            this_NPL[subj_idx]      = deepcopy(fval)
            this_NLPrior[subj_idx]  = deepcopy(nl_prior)
        
        if iiter == 0:
            NPL = deepcopy(this_NPL)
            NLPrior = deepcopy(this_NLPrior)
        else:
            NPL = np.hstack((NPL, this_NPL))
            NLPrior = np.hstack((NLPrior, this_NLPrior))

        ### MAXIMIZATION STEP
        # compute gaussians and sigmas per parameter
        curmu, cursigma, flagcov, _ = compGauss_ms(m,h)

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

    return m, h, posterior, NPL, NLPrior, NLL