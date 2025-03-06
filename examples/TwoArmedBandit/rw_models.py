import numpy as np
from scipy.special import expit
import sys
sys.path.append('../')
from pyEM.math import calc_fval

def softmax(EVs, beta):
    if type(EVs) is list:
        EVs = np.array(EVs)
    return np.exp(beta*EVs) / np.sum(np.exp(beta*EVs))

def norm2beta(x, max_val=20):
    return max_val / (1 + np.exp(-x))

def norm2alpha(x):
    return expit(x)

def simulate(params, nblocks=3, ntrials=24, outcomes=None):
    """
    Simulate the basic RW model.

    Args:
        `params` is a np.array of shape (nsubjects, nparams)
        `nblocks` is the number of blocks to simulate
        `ntrials` is the number of trials per block
    
    Returns:
        `simulated_dict` is a dictionary with the simulated data with the following keys:
            - `choices` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `rewards` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `EV` is a np.array of shape (nsubjects, nblocks, ntrials+1, 2)
            - `CH_PROB` is a np.array of shape (nsubjects, nblocks, ntrials, 2)
            - `CHOICES_A` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `PE` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `CHOICE_NLL` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `params` is a np.array of the parameters used to simulate the data
                - `beta` is the softmax inverse temperature
                - `lr` is the learning rate
    """

    reverse     = 0
    nsubjects   = params.shape[0]
    choices     = np.empty((nsubjects, nblocks, ntrials,), dtype='object')
    rewards     = np.zeros((nsubjects, nblocks, ntrials,))
    EV          = np.zeros((nsubjects, nblocks, ntrials+1, 2))
    CH_PROB     = np.zeros((nsubjects, nblocks, ntrials,   2))
    CHOICES_A   = np.zeros((nsubjects, nblocks, ntrials,))
    PE          = np.zeros((nsubjects, nblocks, ntrials,))
    CHOICE_NLL  = np.zeros((nsubjects, nblocks, ntrials,))

    subj_dict = {}
    this_block_probs = [.8,.2]
    for subj_idx in range(nsubjects):
        beta, lr = params[subj_idx,:]
            
        for b in range(nblocks): # if nblocks == 1, then use reversals
            for t in range(ntrials):
                if nblocks == 1:
                    if (t+1) in [12, 24, 36 ,48, 60, 72, 84, 96, 108, 120]: # reverse
                        this_block_probs = this_block_probs[::-1]
                        if reverse == 1:
                            reverse = 0
                        elif reverse == 0:
                            reverse = 1

                if t == 0:
                    EV[subj_idx, b, t,:]    = [.5,.5]

                # calculate choice probability
                CH_PROB[subj_idx, b, t,:] = softmax(EV[subj_idx, b, t, :], beta)

                # make choice
                choices[subj_idx, b, t]   = np.random.choice(['A', 'B'], 
                                                size=1, 
                                                p=CH_PROB[subj_idx, b, t,:])[0]

                # get choice index
                if choices[subj_idx, b, t] == 'A':
                    c = 0
                    CHOICES_A[subj_idx, b, t] = 1
                    # get outcome
                    if outcomes is None:
                        rewards[subj_idx, b, t]   = np.random.choice([1, 0], 
                                                        size=1, 
                                                        p=this_block_probs)[0]
                    else:
                        rewards[subj_idx, b, t]   = outcomes[b, t, c]
                else:
                    c = 1
                    CHOICES_A[subj_idx, b, t] = 0
                    # get outcome
                    if outcomes is None:
                        rewards[subj_idx, b, t]   = np.random.choice([1, 0], 
                                                        size=1, 
                                                        p=this_block_probs[::-1])[0]
                    else:
                        rewards[subj_idx, b, t]   = outcomes[b, t, c]

                # calculate PE
                PE[subj_idx, b, t] = rewards[subj_idx, b, t] - EV[subj_idx, b, t, c]

                # update EV
                EV[subj_idx, b, t+1, :] = EV[subj_idx, b, t, :].copy()
                EV[subj_idx, b, t+1, c] = EV[subj_idx, b, t, c] + (lr * PE[subj_idx, b, t])
                
                CHOICE_NLL[subj_idx, b, t] = -np.log(CH_PROB[subj_idx, b, t, c])

    # store params
    subj_dict = {'params'    : params,
                 'choices'   : choices, 
                 'rewards'   : rewards, 
                 'EV'        : EV, 
                 'CH_PROB'   : CH_PROB, 
                 'CHOICES_A' : CHOICES_A, 
                 'PE'        : PE, 
                 'CHOICE_NLL': CHOICE_NLL}

    return subj_dict

def fit(params, choices, rewards, prior=None, output='npl'):
    ''' 
    Fit the basic RW model to a single subject's data.
        choices is a np.array with "A" or "B" for each trial
        rewards is a np.array with 1 (cue) or 0 (no cue) for each trial
        output is a string that specifies what to return ('npl', 'nll', or 'all')
    '''
    nparams = len(params)
    beta = norm2beta(params[0]) # transforms to 0 - 20
    lr   = norm2alpha(params[1]) # transforms to 0 - 1

    # make sure params are in range
    this_alpha_bounds = [0, 1]
    if lr < min(this_alpha_bounds) or lr > max(this_alpha_bounds):
        return 10000000
    this_beta_bounds = [0.00001, 20]
    if beta < min(this_beta_bounds) or beta > max(this_beta_bounds):
        return 10000000

    nblocks, ntrials = rewards.shape

    EV          = np.zeros((nblocks, ntrials+1, 2))
    CH_PROB     = np.zeros((nblocks, ntrials,   2))
    CHOICES_A   = np.zeros((nblocks, ntrials,))
    PE          = np.zeros((nblocks, ntrials,))
    CHOICE_NLL  = 0

    for b in range(nblocks):
        for t in range(ntrials):
            if t == 0:
                EV[b, t,:]    = [.5, .5]

            # get choice index
            if choices[b, t] == 'A':
                c = 0
                CHOICES_A[b, t] = 1
            else:
                c = 1
                CHOICES_A[b, t] = 0

            # calculate choice probability
            CH_PROB[b, t,:] = softmax(EV[b, t, :], beta)
            
            # calculate PE
            PE[b, t] = rewards[b, t] - EV[b, t, c]

            # update EV
            EV[b, t+1, :] = EV[b, t, :].copy()
            EV[b, t+1, c] = EV[b, t, c] + (lr * PE[b, t])
            
            # add to sum of choice nll for the block
            CHOICE_NLL += -np.log(CH_PROB[b, t, c])
            
    # CALCULATE NEGATIVE POSTERIOR LIKELIHOOD FROM NEGLL AND PRIOR (OR NEGLL)
    if (output == 'npl') or (output == 'nll'):
        fval = calc_fval(CHOICE_NLL, params, prior=prior, output=output)
        return fval
    
    elif output == 'all':
        subj_dict = {'params'     : [beta, lr],
                     'choices'    : choices, 
                     'rewards'    : rewards, 
                     'EV'         : EV, 
                     'CH_PROB'    : CH_PROB, 
                     'CHOICES_A'  : CHOICES_A, 
                     'PE'         : PE, 
                     'CHOICE_NLL' : CHOICE_NLL,
                     'BIC'        : nparams * np.log(ntrials*nblocks) + 2*CHOICE_NLL}
        return subj_dict