import numpy as np
from tqdm import tqdm
import sys
sys.path.append('../')
from pyEM.math import softmax, norm2beta, norm2alpha

def simulate(params, nblocks=3, ntrials=35, opt_act=None, blocks=None, policy='basic'): #can feed in actual reward result here for outcome
    """
    Simulate the basic/context/valence RW model. 
    Other notes for slot machine: 
    'opt_act' which machine is the best, already encoded reversal: 0-left, 1-right. 
    'reward' whether rewarded or not (context depended), already encoded probabalistic outcome: 0-no, 1-yes. 
    'choice' represents which machine was picked by agent/participant: 0-right, 1-left.

    outcome structre: not exactly 80vs20 probability 
    create your own outcome sequence 
    determine left vs right outcome based on opt_act
    use opt_act for reversal coding 
    assign probaility to opt_action 
    
    use opt_act to determine which one has 80%
    opt_act = 1-right, probabilty[0.2, 0.8] (right is winning machine)
    0-left, probability[0.8, 0.2] (left is winning now)
    L_block_probs = [.8, .2]
    R_block_probs = []

    on each trial:
    determine the L (0) and R (1) outcome based on opt_act
    ```
    opt_act == 0:
    L_this_block_probs = [.8, .2] #make it clean just doing it for Left choice
    else:
    R_this_block_probs = [.2, .8]
    ```
    ```
    if choice[subj_idx, b, t] == 'L':
    rewards[subj_idx, b, t] = np.random.choice([1, 0],
    size=1,
    p=L_blcok_prob[0]
    else
    rewards[subj_idx, b, t] = np.random.choice([1, 0],
    size=1,
    p=R_block_prob([0])
    ```

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
    # initializing data structure 
    # reverse     = 0
    nsubjects   = params.shape[0]
    ev          = np.zeros((nsubjects, nblocks, ntrials+1, 2))
    ch_prob     = np.zeros((nsubjects, nblocks, ntrials,   2))
    choices     = np.empty((nsubjects, nblocks, ntrials,), dtype='object')
    choices_L   = np.zeros((nsubjects, nblocks, ntrials,))
    outcomes    = np.zeros((nsubjects, nblocks, ntrials,))
    pe          = np.zeros((nsubjects, nblocks, ntrials,))
    choice_nll  = np.zeros((nsubjects, nblocks, ntrials,))

    subj_dict = {}
    # this_block_probs = [.8,.2] #slot probabilities

    for subj_idx in tqdm(range(nsubjects)):
        # set params basic on models 
        if policy == 'basic':
            lr, beta = params[subj_idx,:]
        if policy == 'context':
            lr_opt, lr_pes, lr_mix, beta = params[subj_idx, :]
        if policy == 'valence':
            lr_pos, lr_neg, beta = params[subj_idx, :]
        if policy == 'outcome':
            lr_rew, lr_pun, beta = params[subj_idx, :]
            
        for b in range(nblocks): # if nblocks == 1, then use reversals
            for t in range(ntrials):
                # if nblocks == 1: #block dependent reversal coding here. 
                #     if (t+1) in [12, 24, 36 ,48, 60, 72, 84, 96, 108, 120]: # reverse
                #         this_block_probs = this_block_probs[::-1]
                #         if reverse == 1:
                #             reverse = 0
                #         elif reverse == 0:
                #             reverse = 1

                if opt_act[subj_idx, b, t]  == 0: # picking the left is optimal [0, 1] = [left, right]
                    L_block_probs = [.8, .2]      # [win, lost]
                    R_block_probs = [.2, .8]
                else:                             # opt_act == 1 picking right is optimal 
                    L_block_probs = [.2, .8]      # [win, lost]
                    R_block_probs = [.8, .2]

                if t == 0:
                    ev[subj_idx, b, t,:]    = [0.0, 0.0] # initialize expected value

                # calculate choice probability
                ch_prob[subj_idx, b, t,:] = softmax(ev[subj_idx, b, t, :], beta)
                # print(ch_prob[subj_idx, b, t,:])

                choices[subj_idx, b, t]   = np.random.choice([0, 1], # choice: 0-right, 1-left
                                                size=1, 
                                                p=ch_prob[subj_idx, b, t,:])[0]
                print(ch_prob[subj_idx, b, t,:], ev[subj_idx, b, t, :], choices[subj_idx, b, t])
                # sanity check to always pick Left
                # choices[subj_idx, b, t] = 1

                # choice  0=right, 1=left
                # opt_act 0=left,  1=right
                # logic flow on the left machine
                if blocks[subj_idx, b, t] == 'numberbar_neg':                          #in punishment block
                    if choices[subj_idx, b, t] == 1: #if agent pick left machine. first index is rewarded more frequently
                        c = 1
                        choices_L[subj_idx, b, t] = 1   
                        outcomes[subj_idx, b, t] = np.random.choice([0, -1], #this order matters = ['win', 'lost'] block dependent
                        size=1,
                        p=L_block_probs)[0]
                    else:
                        c = 0
                        choices_L[subj_idx, b, t] = 0  
                        outcomes[subj_idx, b, t] = np.random.choice([0, -1],
                        size=1,
                        p=R_block_probs)[0]
                
                if blocks[subj_idx, b, t] == 'numberbar_mixed':                          #in mic block
                    if choices[subj_idx, b, t] == 1:
                        c = 1
                        choices_L[subj_idx, b, t] = 1  
                        outcomes[subj_idx, b, t] = np.random.choice([1, -1],
                        size=1,
                        p=L_block_probs)[0]
                    else:
                        c = 0
                        choices_L[subj_idx, b, t] = 0
                        outcomes[subj_idx, b, t] = np.random.choice([1, -1],
                        size=1,
                        p=R_block_probs)[0]
                
                if blocks[subj_idx, b, t] == 'numberbar_pos':                          #in reward block
                    if choices[subj_idx, b, t] == 1:
                        c = 1
                        choices_L[subj_idx, b, t] = 1  
                        outcomes[subj_idx, b, t] = np.random.choice([1, 0],
                        size=1,
                        p=L_block_probs)[0]
                    else:
                        c = 0
                        choices_L[subj_idx, b, t] = 0
                        outcomes[subj_idx, b, t] = np.random.choice([1, 0],
                        size=1,
                        p=R_block_probs)[0]


                # if choices[subj_idx, b, t] == 1: #pick the left machine 
                #     c = 1
                #     choices_L[subj_idx, b, t] = 1
                #     # get outcome
                #     if outcomes is None:
                #         rewards[subj_idx, b, t]   = np.random.choice([1, 0], 
                #                                         size=1, 
                #                                         p=this_block_probs)[0] # does this return for the better machine of 80% reward? 
                #     else: #DOUBLE CHECK HERE
                #         rewards[subj_idx, b, t]   = outcomes[subj_idx][b][t] #feeding in the reward sequence from data (0-no reward, 1-reward)
                # else:
                #     c = 0
                #     choices_L[subj_idx, b, t] = 0
                #     # get outcome
                #     if outcomes is None:
                #         rewards[subj_idx, b, t]   = np.random.choice([1, 0], 
                #                                         size=1, 
                #                                         p=this_block_probs[::-1])[0] #double check what probability is here for??? 
                #     else:
                #         rewards[subj_idx, b, t]   = outcomes[subj_idx][b][t] #feeding in the reward sequence from data (0-no reward, 1-reward)

                # calculate PE
                pe[subj_idx, b, t] = outcomes[subj_idx, b, t] - ev[subj_idx, b, t, c]

                # update EV (this is model dependent EV update) need to modify for all models. 
                # update rule copied over from model fitting update rules
                # this part is messing up things. (this part debug)
                # double check outcome matching. probably the indexing (for location and update)
                if policy == 'basic':
                    ev[subj_idx, b, t+1, :] = ev[subj_idx, b, t, :].copy()
                    ev[subj_idx, b, t+1, c] = ev[subj_idx, b, t, c] + (lr * pe[subj_idx, b, t])
                
                elif policy == 'context':
                    if blocks[subj_idx, b, t]    == 'numberbar_pos':                          #if optimistic context
                        ev[subj_idx, b, t+1, :] = ev[subj_idx, b, t, :].copy()
                        ev[subj_idx, b, t+1, c] = ev[subj_idx, b, t, c] + (lr_opt * pe[subj_idx, b, t])
                    elif blocks[subj_idx, b, t]  == 'numberbar_neg':                          #if perssimistic context
                        ev[subj_idx, b, t+1, :] = ev[subj_idx, b, t, :].copy()
                        ev[subj_idx, b, t+1, c] = ev[subj_idx, b, t, c] + (lr_pes * pe[subj_idx, b, t])
                    elif blocks[subj_idx, b, t]  == 'numberbar_mixed': 
                        ev[subj_idx, b, t+1, :] = ev[subj_idx, b, t, :].copy()
                        ev[subj_idx, b, t+1, c] = ev[subj_idx, b, t, c] + (lr_mix * pe[subj_idx, b, t])
                
                elif policy == 'outcome':
                    if outcomes[subj_idx, b, t] == 1:                             #if rewarded trial
                        ev[subj_idx, b, t+1, :] = ev[subj_idx, b, t, :].copy()
                        ev[subj_idx, b, t+1, c] = ev[subj_idx, b, t, c] + (lr_rew * pe[subj_idx, b, t])
                    elif outcomes[subj_idx, b, t] == -1:                          #if punishment trial
                        ev[subj_idx, b, t+1, :] = ev[subj_idx, b, t, :].copy()
                        ev[subj_idx, b, t+1, c] = ev[subj_idx, b, t, c] + (lr_pun * pe[subj_idx, b, t])
                    elif outcomes[subj_idx, b, t] == 0 and blocks[subj_idx, b, t] == 'numberbar_neg':  
                        ev[subj_idx, b, t+1, :] = ev[subj_idx, b, t, :].copy()
                        ev[subj_idx, b, t+1, c] = ev[subj_idx, b, t, c] + (lr_rew * pe[subj_idx, b, t])
                    elif outcomes[subj_idx, b, t] == 0 and blocks[subj_idx, b, t] == 'numberbar_pos':  
                        ev[subj_idx, b, t+1, :] = ev[subj_idx, b, t, :].copy()
                        ev[subj_idx, b, t+1, c] = ev[subj_idx, b, t, c] + (lr_pun * pe[subj_idx, b, t])
                
                elif policy == 'valence':
                    if pe[subj_idx, b, t] >= 0:                             #if positive prediction error
                        ev[subj_idx, b, t+1, :] = ev[subj_idx, b, t, :].copy()
                        ev[subj_idx, b, t+1, c] = ev[subj_idx, b, t, c] + (lr_pos * pe[subj_idx, b, t])
                    elif pe[subj_idx, b, t] < 0:                            #if negative prediction error 
                        ev[subj_idx, b, t+1, :] = ev[subj_idx, b, t, :].copy()
                        ev[subj_idx, b, t+1, c] = ev[subj_idx, b, t, c] + (lr_neg * pe[subj_idx, b, t])

                choice_nll[subj_idx, b, t] = ch_prob[subj_idx, b, t, c].copy()

    # store params
    subj_dict = {'params'    : params,
                 'ev'        : ev, 
                 'ch_prob'   : ch_prob, 
                 'choices'   : choices, 
                 'choices_L' : choices_L, 
                 'outcomes'  : outcomes, 
                 'opt_act'   : opt_act,
                 'blocks'    : blocks,
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
    
    if output == 'npl':
        if prior is not None:  # EM-fit: P(Choices | h) * P(h | O) should be maximised, therefore same as minimizing it with negative sign
            fval = -(-negll + prior['logpdf'](params))

            if any(prior['sigma'] == 0):
                this_mu = prior['mu']
                this_sigma = prior['sigma']
                this_logprior = prior['logpdf'](params)
                print(f'mu: {this_mu}')
                print(f'sigma: {this_sigma}')
                print(f'logpdf: {this_logprior}')
                print(f'fval: {fval}')
            
            if np.isinf(fval):
                fval = 10000000
            return fval
        else: # NLL fit 
            return negll
        
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

def fit_slot_context(params, choices, outcomes, blocks, prior=None, output='npl'):
    ''' 
    Fit the basic RW model to a single subject's data.
        choices is a np.array with 1 or 0 for each trial choosing the left slot machine
        outcomes is a np.array with 1 (reward)  0 (no) or -1 (punishment) for each trial
        blocks is a np.array with string for block type in each trial
        output is a string that specifies what to return (either 'nll' or 'all')
    '''
    nparams = len(params)
    lr_opt = norm2alpha(params[0])
    lr_pes = norm2alpha(params[1])
    lr_mix = norm2alpha(params[2])
    beta = norm2beta(params[3])

    # make sure params are in range
    this_alpha_bounds = [0, 1]
    if lr_opt < min(this_alpha_bounds) or lr_opt > max(this_alpha_bounds):
        # print(f'lr = {i_alpha:.3f} not in range')
        return 10000000
    this_alpha_bounds = [0, 1]
    if lr_pes < min(this_alpha_bounds) or lr_pes > max(this_alpha_bounds):
        # print(f'lr = {i_alpha:.3f} not in range')
        return 10000000
    this_alpha_bounds = [0, 1]
    if lr_mix < min(this_alpha_bounds) or lr_mix > max(this_alpha_bounds):
        # print(f'lr = {i_alpha:.3f} not in range')
        return 10000000
    this_beta_bounds = [0.00001, 10]
    if beta < min(this_beta_bounds) or beta > max(this_beta_bounds):
        # print(f'beta = {beta:.3f} not in range')
        return 10000000

    nblocks, ntrials = outcomes.shape 
    # slot: 3 blocks, 35 trials

    ev          = np.zeros((nblocks, ntrials+1, 2))
    ch_prob     = np.zeros((nblocks, ntrials,   2))
    choices_L   = np.zeros((nblocks, ntrials,))
    pe          = np.zeros((nblocks, ntrials,))
    choice_nll  = 0

    for b in range(nblocks): #if nblocks==1, use reversals
        for t in range(ntrials):
            if t == 0:
                ev[b, t,:] = [0, 0] # for this 3 block design task, initiate to 0

            # get choice index
            if choices[b, t] == 1: 
                c = 1
                choices_L[b, t] = 1 # choose the left slot machine (participant choice)
                # choice encoding is consistent with block and reversal. 
            else:
                c = 0
                choices_L[b, t] = 0 # choose the right slot machine (participant choice)

            # calculate choice probability
            ch_prob[b, t,:] = softmax(ev[b, t, :], beta)
            
            # calculate PE
            pe[b, t] = outcomes[b, t] - ev[b, t, c]

            # update EV using outcome-based RL model (see reference)
            # don't hardcode your parameter
            if blocks[b, t]   == 'numberbar_pos':                          #if optimistic context
                ev[b, t+1, :] = ev[b, t, :].copy()
                ev[b, t+1, c] = ev[b, t, c] + (lr_opt * pe[b, t])
            elif blocks[b, t] == 'numberbar_neg':                          #if perssimistic context
                ev[b, t+1, :] = ev[b, t, :].copy()
                ev[b, t+1, c] = ev[b, t, c] + (lr_pes * pe[b, t])
            elif blocks[b, t] == 'numberbar_mixed': 
                ev[b, t+1, :] = ev[b, t, :].copy()
                ev[b, t+1, c] = ev[b, t, c] + (lr_mix * pe[b, t])

            # add to sum of choice nll for the block
            choice_nll += -np.log(ch_prob[b, t, c])
        
    # get the total negative log likelihood
    negll = choice_nll
    
    if output == 'npl':
        if prior is not None:  # EM-fit: P(Choices | h) * P(h | O) should be maximised, therefore same as minimizing it with negative sign
            fval = -(-negll + prior['logpdf'](params))

            if any(prior['sigma'] == 0):
                this_mu = prior['mu']
                this_sigma = prior['sigma']
                this_logprior = prior['logpdf'](params)
                print(f'mu: {this_mu}')
                print(f'sigma: {this_sigma}')
                print(f'logpdf: {this_logprior}')
                print(f'fval: {fval}')
            
            if np.isinf(fval):
                fval = 10000000
            return fval
        else: # NLL fit 
            return negll
        
    elif output == 'all':
        subj_dict = {'params'     : [lr_opt, lr_pes, lr_mix, beta],
                     'ev'         : ev, 
                     'ch_prob'    : ch_prob, 
                     'choices'    : choices, 
                     'choices_L'  : choices_L,
                     'blocks'     : blocks,
                     'outcomes'   : outcomes, 
                     'pe'         : pe, 
                     'negll'      : negll,
                     'BIC'        : nparams * np.log(ntrials*nblocks) + 2*negll}
        return subj_dict

def fit_slot_context_r(params, choices, outcomes, blocks, prior=None, output='npl'):
    ''' 
    Fit the basic RW model to a single subject's data.
        choices is a np.array with 1 or 0 for each trial choosing the left slot machine
        outcomes is a np.array with 1 (reward)  0 (no) or -1 (punishment) for each trial
        blocks is a np.array with string for block type in each trial
        output is a string that specifies what to return (either 'nll' or 'all')
    '''
    nparams = len(params)
    lr_opt = norm2alpha(params[0])
    lr_pes = norm2alpha(params[1])
    beta = norm2beta(params[2])

    # make sure params are in range
    this_alpha_bounds = [0, 1]
    if lr_opt < min(this_alpha_bounds) or lr_opt > max(this_alpha_bounds):
        # print(f'lr = {i_alpha:.3f} not in range')
        return 10000000
    this_alpha_bounds = [0, 1]
    if lr_pes < min(this_alpha_bounds) or lr_pes > max(this_alpha_bounds):
        # print(f'lr = {i_alpha:.3f} not in range')
        return 10000000
    this_beta_bounds = [0.00001, 10]
    if beta < min(this_beta_bounds) or beta > max(this_beta_bounds):
        # print(f'beta = {beta:.3f} not in range')
        return 10000000

    nblocks, ntrials = outcomes.shape 
    # slot: 3 blocks, 35 trials

    ev          = np.zeros((nblocks, ntrials+1, 2))
    ch_prob     = np.zeros((nblocks, ntrials,   2))
    choices_L   = np.zeros((nblocks, ntrials,))
    pe          = np.zeros((nblocks, ntrials,))
    choice_nll  = 0

    for b in range(nblocks): #if nblocks==1, use reversals
        for t in range(ntrials):
            if t == 0:
                ev[b, t,:] = [0, 0] # for this 3 block design task, initiate to 0

            # get choice index
            if choices[b, t] == 1: 
                c = 1
                choices_L[b, t] = 1 # choose the left slot machine (participant choice)
                # choice encoding is consistent with block and reversal. 
            else:
                c = 0
                choices_L[b, t] = 0 # choose the right slot machine (participant choice)

            # calculate choice probability
            ch_prob[b, t,:] = softmax(ev[b, t, :], beta)
            
            # calculate PE
            pe[b, t] = outcomes[b, t] - ev[b, t, c]

            # update EV using outcome-based RL model (see reference)
            # don't hardcode your parameter
            if blocks[b, t]   == 'numberbar_pos':                          #if optimistic context
                ev[b, t+1, :] = ev[b, t, :].copy()
                ev[b, t+1, c] = ev[b, t, c] + (lr_opt * pe[b, t])
            elif blocks[b, t] == 'numberbar_neg':                          #if perssimistic context
                ev[b, t+1, :] = ev[b, t, :].copy()
                ev[b, t+1, c] = ev[b, t, c] + (lr_pes * pe[b, t])
            elif blocks[b, t] == 'numberbar_mixed': 
                ev[b, t+1, :] = ev[b, t, :].copy()                      #if mixed context with ratio between opt and pes lr
                ev[b, t+1, c] = ev[b, t, c] + ((lr_opt/lr_pes) * pe[b, t])

            # add to sum of choice nll for the block
            choice_nll += -np.log(ch_prob[b, t, c])
        
    # get the total negative log likelihood
    negll = choice_nll
    
    if output == 'npl':
        if prior is not None:  # EM-fit: P(Choices | h) * P(h | O) should be maximised, therefore same as minimizing it with negative sign
            fval = -(-negll + prior['logpdf'](params))

            if any(prior['sigma'] == 0):
                this_mu = prior['mu']
                this_sigma = prior['sigma']
                this_logprior = prior['logpdf'](params)
                print(f'mu: {this_mu}')
                print(f'sigma: {this_sigma}')
                print(f'logpdf: {this_logprior}')
                print(f'fval: {fval}')
            
            if np.isinf(fval):
                fval = 10000000
            return fval
        else: # NLL fit 
            return negll
        
    elif output == 'all':
        subj_dict = {'params'     : [lr_opt, lr_pes, beta],
                     'ev'         : ev, 
                     'ch_prob'    : ch_prob, 
                     'choices'    : choices, 
                     'choices_L'  : choices_L,
                     'blocks'     : blocks,
                     'outcomes'   : outcomes, 
                     'pe'         : pe, 
                     'negll'      : negll,
                     'BIC'        : nparams * np.log(ntrials*nblocks) + 2*negll}
        return subj_dict


def fit_slot_outcome(params, choices, outcomes, blocks, prior=None, output='npl'):
    ''' 
    Fit the basic RW model to a single subject's data.
        choices is a np.array with 1 or 0 for each trial choosing the left slot machine
        outcomes is a np.array with 1 (reward)  0 (no) or -1 (punishment) for each trial
        blocks is a np.array with string for block type in each trial
        output is a string that specifies what to return (either 'nll' or 'all')
    '''
    nparams = len(params)
    lr_rew = norm2alpha(params[0])
    lr_pun = norm2alpha(params[1])
    beta = norm2beta(params[2])

    # make sure params are in range
    this_alpha_bounds = [0, 1]
    if lr_rew < min(this_alpha_bounds) or lr_rew > max(this_alpha_bounds):
        # print(f'lr = {i_alpha:.3f} not in range')
        return 10000000
    this_alpha_bounds = [0, 1]
    if lr_pun < min(this_alpha_bounds) or lr_pun > max(this_alpha_bounds):
        # print(f'lr = {i_alpha:.3f} not in range')
        return 10000000
    this_beta_bounds = [0.00001, 10]
    if beta < min(this_beta_bounds) or beta > max(this_beta_bounds):
        # print(f'beta = {beta:.3f} not in range')
        return 10000000

    nblocks, ntrials = outcomes.shape 
    # slot: 3 blocks, 35 trials

    ev          = np.zeros((nblocks, ntrials+1, 2))
    ch_prob     = np.zeros((nblocks, ntrials,   2))
    choices_L   = np.zeros((nblocks, ntrials,))
    pe          = np.zeros((nblocks, ntrials,))
    choice_nll  = 0

    for b in range(nblocks): #if nblocks==1, use reversals
        for t in range(ntrials):
            if t == 0:
                ev[b, t,:] = [0, 0] # for this 3 block design task, initiate to 0

            # get choice index
            if choices[b, t] == 1: 
                c = 1
                choices_L[b, t] = 1 # choose the left slot machine (participant choice)
                # choice encoding is consistent with block and reversal. 
            else:
                c = 0
                choices_L[b, t] = 0 # choose the right slot machine (participant choice)

            # calculate choice probability
            ch_prob[b, t,:] = softmax(ev[b, t, :], beta)
            
            # calculate PE
            pe[b, t] = outcomes[b, t] - ev[b, t, c]

            # update EV using outcome-based RL model (see reference)
            # don't hardcode your parameter
            if outcomes[b, t] == 1:                             #if rewarded trial
                ev[b, t+1, :] = ev[b, t, :].copy()
                ev[b, t+1, c] = ev[b, t, c] + (lr_rew * pe[b, t])
            elif outcomes[b, t] == -1:                          #if punishment trial
                ev[b, t+1, :] = ev[b, t, :].copy()
                ev[b, t+1, c] = ev[b, t, c] + (lr_pun * pe[b, t])
            elif outcomes[b, t] == 0 and blocks[b, t] == 'numberbar_neg':  
                ev[b, t+1, :] = ev[b, t, :].copy()
                ev[b, t+1, c] = ev[b, t, c] + (lr_rew * pe[b, t])
            elif outcomes[b, t] == 0 and blocks[b, t] == 'numberbar_pos':  
                ev[b, t+1, :] = ev[b, t, :].copy()
                ev[b, t+1, c] = ev[b, t, c] + (lr_pun * pe[b, t])
            
            # add to sum of choice nll for the block
            choice_nll += -np.log(ch_prob[b, t, c])
        
    # get the total negative log likelihood
    negll = choice_nll
    
    if output == 'npl':
        if prior is not None:  # EM-fit: P(Choices | h) * P(h | O) should be maximised, therefore same as minimizing it with negative sign
            fval = -(-negll + prior['logpdf'](params))

            if any(prior['sigma'] == 0):
                this_mu = prior['mu']
                this_sigma = prior['sigma']
                this_logprior = prior['logpdf'](params)
                print(f'mu: {this_mu}')
                print(f'sigma: {this_sigma}')
                print(f'logpdf: {this_logprior}')
                print(f'fval: {fval}')
            
            if np.isinf(fval):
                fval = 10000000
            return fval
        else: # NLL fit 
            return negll
        
    elif output == 'all':
        subj_dict = {'params'     : [lr_rew, lr_pun, beta],
                     'ev'         : ev, 
                     'ch_prob'    : ch_prob, 
                     'choices'    : choices, 
                     'choices_L'  : choices_L,
                     'blocks'     : blocks,
                     'outcomes'   : outcomes, 
                     'pe'         : pe, 
                     'negll'      : negll,
                     'BIC'        : nparams * np.log(ntrials*nblocks) + 2*negll}
        return subj_dict

def fit_slot_valence(params, choices, outcomes, blocks, prior=None, output='npl'):
    ''' 
    Fit the basic RW model to a single subject's data.
        choices is a np.array with 1 or 0 for each trial choosing the left slot machine
        outcomes is a np.array with 1 (reward)  0 (no) or -1 (punishment) for each trial
        blocks is a np.array with string for block type in each trial
        output is a string that specifies what to return (either 'nll' or 'all')
    '''
    nparams = len(params)
    lr_pos = norm2alpha(params[0])
    lr_neg = norm2alpha(params[1])
    beta = norm2beta(params[2])

    # make sure params are in range
    this_alpha_bounds = [0, 1]
    if lr_pos < min(this_alpha_bounds) or lr_pos > max(this_alpha_bounds):
        # print(f'lr = {i_alpha:.3f} not in range')
        return 10000000
    this_alpha_bounds = [0, 1]
    if lr_neg < min(this_alpha_bounds) or lr_neg > max(this_alpha_bounds):
        # print(f'lr = {i_alpha:.3f} not in range')
        return 10000000
    this_beta_bounds = [0.00001, 10]
    if beta < min(this_beta_bounds) or beta > max(this_beta_bounds):
        # print(f'beta = {beta:.3f} not in range')
        return 10000000

    nblocks, ntrials = outcomes.shape 
    # slot: 3 blocks, 35 trials

    ev          = np.zeros((nblocks, ntrials+1, 2))
    ch_prob     = np.zeros((nblocks, ntrials,   2))
    choices_L   = np.zeros((nblocks, ntrials,))
    pe          = np.zeros((nblocks, ntrials,))
    choice_nll  = 0

    for b in range(nblocks): 
        for t in range(ntrials):
            if t == 0:
                ev[b, t,:] = [0, 0] # for this 3 block design task, initiate to 0

            # get choice index
            if choices[b, t] == 1: 
                c = 1
                choices_L[b, t] = 1 # choose the left slot machine (participant choice)
                # choice encoding is consistent with block and reversal. 
            else:
                c = 0
                choices_L[b, t] = 0 # choose the right slot machine (participant choice)

            # calculate choice probability
            ch_prob[b, t,:] = softmax(ev[b, t, :], beta)
            
            # calculate PE
            pe[b, t] = outcomes[b, t] - ev[b, t, c]

            # update EV using Prediction error based RL model  
            # don't hardcode your parameter
            if pe[b, t] >= 0:                             #if positive prediction error
                ev[b, t+1, :] = ev[b, t, :].copy()
                ev[b, t+1, c] = ev[b, t, c] + (lr_pos * pe[b, t])
            elif pe[b, t] < 0:                          #if negative prediction error 
                ev[b, t+1, :] = ev[b, t, :].copy()
                ev[b, t+1, c] = ev[b, t, c] + (lr_neg * pe[b, t])
            
            # add to sum of choice nll for the block
            choice_nll += -np.log(ch_prob[b, t, c])
        
    # get the total negative log likelihood
    negll = choice_nll
    
    if output == 'npl':
        if prior is not None:  # EM-fit: P(Choices | h) * P(h | O) should be maximised, therefore same as minimizing it with negative sign
            fval = -(-negll + prior['logpdf'](params))

            if any(prior['sigma'] == 0):
                this_mu = prior['mu']
                this_sigma = prior['sigma']
                this_logprior = prior['logpdf'](params)
                print(f'mu: {this_mu}')
                print(f'sigma: {this_sigma}')
                print(f'logpdf: {this_logprior}')
                print(f'fval: {fval}')
            
            if np.isinf(fval):
                fval = 10000000
            return fval
        else: # NLL fit 
            return negll
        
    elif output == 'all':
        subj_dict = {'params'     : [lr_pos, lr_neg, beta],
                     'ev'         : ev, 
                     'ch_prob'    : ch_prob, 
                     'choices'    : choices, 
                     'choices_L'  : choices_L,
                     'blocks'     : blocks,
                     'outcomes'   : outcomes, 
                     'pe'         : pe, 
                     'negll'      : negll,
                     'BIC'        : nparams * np.log(ntrials*nblocks) + 2*negll}
        return subj_dict

def fit_slot_basic(params, choices, outcomes, blocks, prior=None, output='npl'):
    ''' 
    Fit the basic RW model to a single subject's data.
        choices is a np.array with 1 or 0 for each trial choosing the optimal slot
        rewards is a np.array with 1 (reward) or 0 (no) for each trial [seperate this by block!]
        output is a string that specifies what to return (either 'nll' or 'all')
    '''
    nparams = len(params)
    lr = norm2alpha(params[0])
    beta = norm2beta(params[1])

    # make sure params are in range
    this_alpha_bounds = [0, 1]
    if lr < min(this_alpha_bounds) or lr > max(this_alpha_bounds):
        # print(f'lr = {i_alpha:.3f} not in range')
        return 10000000
    this_beta_bounds = [0.00001, 10]
    if beta < min(this_beta_bounds) or beta > max(this_beta_bounds):
        # print(f'beta = {beta:.3f} not in range')
        return 10000000

    nblocks, ntrials = outcomes.shape
    # 3 blocks, 35 trials

    ev          = np.zeros((nblocks, ntrials+1, 2))
    ch_prob     = np.zeros((nblocks, ntrials,   2))
    choices_L   = np.zeros((nblocks, ntrials,))
    pe          = np.zeros((nblocks, ntrials,))
    choice_nll  = 0

    for b in range(nblocks):
        for t in range(ntrials):
            if t == 0:
                ev[b, t,:] = [0, 0] # for this 3 block design task, initiate to 0

            # get choice index
            if choices[b, t] == 1: 
                c = 1
                choices_L[b, t] = 1 # choose the left slot machine (participant choice)
                # actual slot machine choice double check  (use the action variable, the choice participant make)
                # choice encoding is consistent with block and reversal. 
                # confirm action column: if it always just encode the choice they make
                
            else:
                c = 0
                choices_L[b, t] = 0 # choose the right slot machine (participant choice)

            # calculate choice probability
            ch_prob[b, t,:] = softmax(ev[b, t, :], beta)
            
            # calculate PE
            pe[b, t] = outcomes[b, t] - ev[b, t, c]

            # update EV 
            ev[b, t+1, :] = ev[b, t, :].copy()
            ev[b, t+1, c] = ev[b, t, c] + (lr * pe[b, t])

            # add to sum of choice nll for the block
            choice_nll += -np.log(ch_prob[b, t, c])
        
    # get the total negative log likelihood
    negll = choice_nll
    
    if output == 'npl':
        if prior is not None:  # EM-fit: P(Choices | h) * P(h | O) should be maximised, therefore same as minimizing it with negative sign
            fval = -(-negll + prior['logpdf'](params))

            if any(prior['sigma'] == 0):
                this_mu = prior['mu']
                this_sigma = prior['sigma']
                this_logprior = prior['logpdf'](params)
                print(f'mu: {this_mu}')
                print(f'sigma: {this_sigma}')
                print(f'logpdf: {this_logprior}')
                print(f'fval: {fval}')
            
            if np.isinf(fval):
                fval = 10000000
            return fval
        else: # NLL fit 
            return negll
        
    elif output == 'all':
        subj_dict = {'params'     : [lr, beta],
                     'ev'         : ev, 
                     'ch_prob'    : ch_prob, 
                     'choices'    : choices, 
                     'choices_L'  : choices_L,
                     'blocks'     : blocks,
                     'outcomes'   : outcomes, 
                     'pe'         : pe, 
                     'negll'      : negll,
                     'BIC'        : nparams * np.log(ntrials*nblocks) + 2*negll}
        return subj_dict