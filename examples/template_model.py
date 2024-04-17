import numpy as np, sys
sys.path.append('../')
from pyEM.math import softmax, norm2beta, norm2alpha, calc_fval

def fit(params, data, prior=None, output='npl'):
    ''' 
    This is a toy example `fit` function that can be adapted to implement your own model.
        `params` is a list of parameters to be fit
        `data` can be a np.array with choices, rewards for each trial; shape = (2, ntrials)
            (i.e., choices are "A" or "B"; rewards are 1 or 0)
        `prior` is a dictionary with the mean and variance of the prior distribution of the parameters 
            (this is taken care of in the `expectation_step` function in `pyEM/fitting.py`)
        `output` is a string that specifies what to return (e.g., 'npl', 'nll', or 'all')
            ('npl' should be used for EM+MAP; 'nll' can be used for MLE)
    '''
    # (1) CONVERT TO PARAMETER SPACE
    # Here, we are just using inverse temperature and learning rate as example parameters (see RW model)
    # You can create your own custom function or use the ones provided
    inverse_temp  = norm2beta(params[0]) 
    learning_rate = norm2alpha(params[1])

    # (2) ENSURE PARAMETERS ARE IN BOUNDS
    this_alpha_bounds = [0, 1]
    if learning_rate < min(this_alpha_bounds) or learning_rate > max(this_alpha_bounds):
        return 10000000
    this_beta_bounds = [0.00001, 10]
    if inverse_temp < min(this_beta_bounds) or inverse_temp > max(this_beta_bounds):
        return 10000000
    
    # (3) INITIALIZE VARIABLES
    ntrials = data.shape[-1]
    choice_nll  = 0

    # (4) LOOP THOUGH TRIALS + CALCULATE NEGATIVE LOG-LIKELIHOOD VIA CHOICE PROBABILITY ACCORDING TO YOUR MODEL
    for t in range(ntrials):
        # Here, we will just randomly generate a choice probability to demonstrate the fitting process
        choice_prob = softmax(np.random.rand(), inverse_temp)
        choice_nll += -np.log(choice_prob)
        
    # (5) CALCULATE NEGATIVE POSTERIOR LIKELIHOOD FROM NEGLL AND PRIOR
    fval = calc_fval(choice_nll, prior, params, output)

    return fval