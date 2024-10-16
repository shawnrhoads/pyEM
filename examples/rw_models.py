import numpy as np
from tqdm import tqdm
import sys
sys.path.append('../')
from pyEM.math import softmax, norm2beta, norm2alpha, calc_fval

def simulate(params, nblocks=3, ntrials=24, outcomes=None):
    """
    Simulate the basic RW model.

    Args:
        `params` is a np.array of shape (nsubjects, nparams)
        `nblocks` is the number of blocks to simulate
        `ntrials` is the number of trials per block
    
    Returns:
        `simulated_dict` is a dictionary with the simulated data with the following keys:
            - `ev` is a np.array of shape (nsubjects, nblocks, ntrials+1, 2)
            - `ch_prob` is a np.array of shape (nsubjects, nblocks, ntrials, 2)
            - `choices` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `choices_A` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `rewards` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `pe` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `choice_nll` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `params` is a np.array of the parameters used to simulate the data
                - `beta` is the softmax inverse temperature
                - `lr` is the learning rate
    """

    reverse     = 0
    nsubjects   = params.shape[0]
    ev          = np.zeros((nsubjects, nblocks, ntrials+1, 2))
    ch_prob     = np.zeros((nsubjects, nblocks, ntrials,   2))
    choices     = np.empty((nsubjects, nblocks, ntrials,), dtype='object')
    choices_A   = np.zeros((nsubjects, nblocks, ntrials,))
    rewards     = np.zeros((nsubjects, nblocks, ntrials,))
    pe          = np.zeros((nsubjects, nblocks, ntrials,))
    choice_nll  = np.zeros((nsubjects, nblocks, ntrials,))

    subj_dict = {}
    this_block_probs = [.8,.2]
    for subj_idx in tqdm(range(nsubjects)):
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
                    ev[subj_idx, b, t,:]    = [.5,.5]

                # calculate choice probability
                ch_prob[subj_idx, b, t,:] = softmax(ev[subj_idx, b, t, :], beta)

                # make choice
                choices[subj_idx, b, t]   = np.random.choice(['A', 'B'], 
                                                size=1, 
                                                p=ch_prob[subj_idx, b, t,:])[0]

                # get choice index
                if choices[subj_idx, b, t] == 'A':
                    c = 0
                    choices_A[subj_idx, b, t] = 1
                    # get outcome
                    if outcomes is None:
                        rewards[subj_idx, b, t]   = np.random.choice([1, 0], 
                                                        size=1, 
                                                        p=this_block_probs)[0]
                    else:
                        rewards[subj_idx, b, t]   = outcomes[b, t, c]
                else:
                    c = 1
                    choices_A[subj_idx, b, t] = 0
                    # get outcome
                    if outcomes is None:
                        rewards[subj_idx, b, t]   = np.random.choice([1, 0], 
                                                        size=1, 
                                                        p=this_block_probs[::-1])[0]
                    else:
                        rewards[subj_idx, b, t]   = outcomes[b, t, c]

                # calculate PE
                pe[subj_idx, b, t] = rewards[subj_idx, b, t] - ev[subj_idx, b, t, c]

                # update EV
                ev[subj_idx, b, t+1, :] = ev[subj_idx, b, t, :].copy()
                ev[subj_idx, b, t+1, c] = ev[subj_idx, b, t, c] + (lr * pe[subj_idx, b, t])
                
                choice_nll[subj_idx, b, t] = -np.log(ch_prob[subj_idx, b, t, c])

    # store params
    subj_dict = {'params'    : params,
                 'ev'        : ev, 
                 'ch_prob'   : ch_prob, 
                 'choices'   : choices, 
                 'choices_A' : choices_A, 
                 'rewards'   : rewards, 
                 'pe'        : pe, 
                 'choice_nll': choice_nll}

    return subj_dict

def fit(params, choices, rewards, prior=None, output='npl'):
    ''' 
    Fit the basic RW model to a single subject's data.
        choices is a np.array with "A" or "B" for each trial
        rewards is a np.array with 1 (cue) or 0 (no cue) for each trial
        output is a string that specifies what to return (either 'nll' or 'all')
    '''
    nparams = len(params)
    beta = norm2beta(params[0])
    lr   = norm2alpha(params[1])

    # make sure params are in range
    this_alpha_bounds = [0, 1]
    if lr < min(this_alpha_bounds) or lr > max(this_alpha_bounds):
        # print(f'lr = {i_alpha:.3f} not in range')
        return 10000000
    this_beta_bounds = [0.00001, 10]
    if beta < min(this_beta_bounds) or beta > max(this_beta_bounds):
        # print(f'beta = {beta:.3f} not in range')
        return 10000000

    nblocks, ntrials = rewards.shape

    ev          = np.zeros((nblocks, ntrials+1, 2))
    ch_prob     = np.zeros((nblocks, ntrials,   2))
    choices_A   = np.zeros((nblocks, ntrials,))
    pe          = np.zeros((nblocks, ntrials,))
    choice_nll  = 0

    for b in range(nblocks): #if nblocks==1, use reversals
        for t in range(ntrials):
            if t == 0:
                ev[b, t,:]    = [.5, .5]

            # get choice index
            if choices[b, t] == 'A':
                c = 0
                choices_A[b, t] = 1
            else:
                c = 1
                choices_A[b, t] = 0

            # calculate choice probability
            ch_prob[b, t,:] = softmax(ev[b, t, :], beta)
            
            # calculate PE
            pe[b, t] = rewards[b, t] - ev[b, t, c]

            # update EV
            ev[b, t+1, :] = ev[b, t, :].copy()
            ev[b, t+1, c] = ev[b, t, c] + (lr * pe[b, t])
            
            # add to sum of choice nll for the block
            choice_nll += -np.log(ch_prob[b, t, c])
        
    # get the total negative log likelihood
    negll = choice_nll
        
    # CALCULATE NEGATIVE POSTERIOR LIKELIHOOD FROM NEGLL AND PRIOR (OR NEGLL)
    if (output == 'npl') or (output == 'nll'):
        fval = calc_fval(choice_nll, params, prior=prior, output=output)
        return fval
    
    elif output == 'all':
        subj_dict = {'params'     : [beta, lr],
                     'ev'         : ev, 
                     'ch_prob'    : ch_prob, 
                     'choices'    : choices, 
                     'choices_A'  : choices_A, 
                     'rewards'    : rewards, 
                     'pe'         : pe, 
                     'negll'      : negll,
                     'BIC'        : nparams * np.log(ntrials*nblocks) + 2*negll}
        return subj_dict