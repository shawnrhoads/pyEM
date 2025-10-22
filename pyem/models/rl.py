
import numpy as np, random
from itertools import permutations, chain
from ..utils.math import softmax, norm2alpha, norm2beta, calc_fval

def rw1a1b_sim(params: np.ndarray, nblocks: int = 3, ntrials: int = 24,
                    outcomes: np.ndarray | None = None):
    """Simulate a simple Rescorla–Wagner model with one learning rate.

    Each subject repeatedly chooses between two options (A/B).  Rewards are
    generated from Bernoulli distributions whose probabilities switch between
    blocks.  Parameters are expected in **Gaussian** space and are transformed
    to their natural ranges via :func:`norm2beta` (for inverse temperature) and
    :func:`norm2alpha` (for learning rate).
    """
    nsubjects = params.shape[0]
    # preallocate arrays for speed
    choices = np.empty((nsubjects, nblocks, ntrials), dtype=object)
    rewards = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    EV = np.zeros((nsubjects, nblocks, ntrials + 1, 2), dtype=float)  # expected values
    ch_prob = np.zeros((nsubjects, nblocks, ntrials, 2), dtype=float)  # choice probabilities
    choices_A = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    PE = np.zeros((nsubjects, nblocks, ntrials), dtype=float)  # prediction errors
    nll = np.zeros((nsubjects, nblocks, ntrials), dtype=float)

    rng = np.random.default_rng()
    this_block_probs = np.array([0.8, 0.2])  # reward probability for option A

    # all params (assuming raw params)
    all_beta = params[:, 0]
    all_alpha = params[:, 1]
    
    # bounds checks
    if not ((all_beta >= 1e-5) & (all_beta <= 20.0)).all():
        raise ValueError("Beta values out of bounds")
    if not ((all_alpha >= 0.0)  & (all_alpha <= 1.0)).all():
        raise ValueError("Alpha values out of bounds")

    for s in range(nsubjects):
        beta = float(all_beta[s])
        alpha = float(all_alpha[s])
        for b in range(nblocks):
            EV[s, b, 0, :] = 0.5
            for t in range(ntrials):
                # softmax translates EVs into choice probabilities
                p = softmax(EV[s, b, t, :], beta)
                ch_prob[s, b, t, :] = p
                c = rng.choice([0, 1], p=p)  # sample a choice
                choices[s, b, t] = "A" if c == 0 else "B"
                choices_A[s, b, t] = 1.0 if c == 0 else 0.0
                if outcomes is None:
                    # reward contingent on chosen option
                    rew = rng.choice([1.0, 0.0], p=this_block_probs if c == 0 else this_block_probs[::-1])
                else:
                    rew = float(outcomes[b, t, c])
                rewards[s, b, t] = rew
                PE[s, b, t] = rew - EV[s, b, t, c]
                # update only the chosen option
                EV[s, b, t + 1, :] = EV[s, b, t, :]
                EV[s, b, t + 1, c] = EV[s, b, t, c] + alpha * PE[s, b, t]
                nll[s, b, t] = -np.log(p[c] + 1e-12)

    return {
        "params": np.array([all_beta, all_alpha]).T,
        "choices": choices,
        "rewards": rewards,
        "EV": EV,
        "ch_prob": ch_prob,
        "choices_A": choices_A,
        "PE": PE,
        "nll": nll,
    }

def rw1a1b_fit(params, choices, rewards, prior=None, output="npl"):
    """
    A thin adapter compatible with EM: returns NPL or NLL.
    params: (2,) in normalized space
    """
    beta = float(norm2beta(params[0]))
    alpha = float(norm2alpha(params[1]))

    # reject values outside natural bounds
    if not (1e-5 <= beta <= 20.0):
        return 1e7
    if not (0.0 <= alpha <= 1.0):
        return 1e7

    nblocks, ntrials = rewards.shape
    EV = np.zeros((nblocks, ntrials + 1, 2))
    PE = np.zeros((nblocks, ntrials))
    nll = 0.0
    for b in range(nblocks):
        EV[b, 0, :] = 0.5
        for t in range(ntrials):
            c = 0 if choices[b, t] == "A" else 1
            p = softmax(EV[b, t, :], beta)
            r = rewards[b, t]
            PE[b, t] = r - EV[b, t, c]
            EV[b, t + 1, :] = EV[b, t, :]
            EV[b, t + 1, c] = EV[b, t, c] + alpha * PE[b, t]
            nll += -np.log(p[c] + 1e-12)

    if output == "all":
        choices_A = (np.asarray(choices) == "A").astype(float)
        subj_dict = {
            'params'   : [beta, alpha],
            'choices'  : choices,
            'choices_A': choices_A,
            'rewards'  : rewards,
            'EV'       : EV,
            'PE'       : PE,
            'nll'      : nll,
        }
        return subj_dict

    # otherwise compute objective value
    return calc_fval(nll, params, prior=prior, output=output)


def rw2a1b_sim(params: np.ndarray, nblocks: int = 3, ntrials: int = 24,
               outcomes: np.ndarray | None = None):
    """Simulate a Rescorla–Wagner model with separate learning rates for gains and losses."""
    nsubjects = params.shape[0]
    choices = np.empty((nsubjects, nblocks, ntrials), dtype=object)
    rewards = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    EV = np.zeros((nsubjects, nblocks, ntrials + 1, 2), dtype=float)
    ch_prob = np.zeros((nsubjects, nblocks, ntrials, 2), dtype=float)
    choices_A = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    PE = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    nll = np.zeros((nsubjects, nblocks, ntrials), dtype=float)

    rng = np.random.default_rng()
    this_block_probs = np.array([0.8, 0.2])

    # all params (assuming raw params)
    all_beta = params[:, 0]
    all_alpha_pos = params[:, 1]
    all_alpha_neg = params[:, 2]
    
    # bounds checks
    if not ((all_beta >= 1e-5) & (all_beta <= 20.0)).all():
        raise ValueError("Beta values out of bounds")
    if not ((all_alpha_pos >= 0.0)  & (all_alpha_pos <= 1.0)).all():
        raise ValueError("Alpha_pos values out of bounds")
    if not ((all_alpha_neg >= 0.0)  & (all_alpha_neg <= 1.0)).all():
        raise ValueError("Alpha_neg values out of bounds")

    for s in range(nsubjects):
        beta = float(all_beta[s])
        alpha_pos = float(all_alpha_pos[s])
        alpha_neg = float(all_alpha_neg[s])
        for b in range(nblocks):
            EV[s, b, 0, :] = 0.5
            for t in range(ntrials):
                p = softmax(EV[s, b, t, :], beta)
                ch_prob[s, b, t, :] = p
                c = rng.choice([0, 1], p=p)
                choices[s, b, t] = "A" if c == 0 else "B"
                choices_A[s, b, t] = 1.0 if c == 0 else 0.0
                if outcomes is None:
                    rew = rng.choice([1.0, 0.0], p=this_block_probs if c == 0 else this_block_probs[::-1])
                else:
                    rew = float(outcomes[b, t, c])
                rewards[s, b, t] = rew
                PE[s, b, t] = rew - EV[s, b, t, c]
                EV[s, b, t + 1, :] = EV[s, b, t, :]
                # apply positive or negative learning rate
                if PE[s, b, t] >= 0:
                    EV[s, b, t + 1, c] = EV[s, b, t, c] + alpha_pos * PE[s, b, t]
                else:
                    EV[s, b, t + 1, c] = EV[s, b, t, c] + alpha_neg * PE[s, b, t]
                nll[s, b, t] = -np.log(p[c] + 1e-12)

    return {
        "params": np.array([all_beta, all_alpha_pos, all_alpha_neg]).T,
        "choices": choices,
        "rewards": rewards,
        "EV": EV,
        "ch_prob": ch_prob,
        "choices_A": choices_A,
        "PE": PE,
        "nll": nll,
    }

def rw2a1b_fit(params, choices, rewards, prior=None, output="npl"):
    """
    A thin adapter compatible with EM: returns NPL or NLL.
    params: (3,) in normalized space
    """
    beta = float(norm2beta(params[0]))
    alpha_pos = float(norm2alpha(params[1]))
    alpha_neg = float(norm2alpha(params[2]))

    # bounds checks
    if not (1e-5 <= beta <= 20.0):
        return 1e7
    if not (0.0 <= alpha_pos <= 1.0):
        return 1e7
    if not (0.0 <= alpha_neg <= 1.0):
        return 1e7

    nblocks, ntrials = rewards.shape
    EV = np.zeros((nblocks, ntrials + 1, 2))
    PE = np.zeros((nblocks, ntrials))
    nll = 0.0
    for b in range(nblocks):
        EV[b, 0, :] = 0.5
        for t in range(ntrials):
            c = 0 if choices[b, t] == "A" else 1
            p = softmax(EV[b, t, :], beta)
            r = rewards[b, t]
            PE[b, t] = r - EV[b, t, c]
            EV[b, t + 1, :] = EV[b, t, :]
            if PE[b, t] > 0:
                EV[b, t + 1, c] = EV[b, t, c] + alpha_pos * PE[b, t]
            else:
                EV[b, t + 1, c] = EV[b, t, c] + alpha_neg * PE[b, t]
            nll += -np.log(p[c] + 1e-12)

    if output == "all":
        choices_A = (np.asarray(choices) == "A").astype(float)
        subj_dict = {
            'params'   : [beta, alpha_pos, alpha_neg],
            'choices'  : choices,
            'rewards'  : rewards,
            'choices_A': choices_A,
            'EV'       : EV,
            'PE'       : PE,
            'nll'      : nll,
        }
        return subj_dict

    return calc_fval(nll, params, prior=prior, output=output)

# --------------------------------------
# 3α-1β SIMULATE (Lockwood et al., 2016)
# --------------------------------------
def rw3a1b_sim(params: np.ndarray,
               nblocks: int = 9,
               ntrials: int = 16,
               outcomes: np.ndarray | None = None):
    """
    Two-option task (A/B each trial) with three binary outcome channels:
    self, other, noone. NATURAL-SPACE params.

    params: (S,4) = [beta, a_self, a_other, a_noone], with:
            beta in [1e-5, 20], alphas in [0,1].
    outcomes (optional): (B,T,2,3) last dim = [self, other, noone]; values in {0,1}.

    Returns:
      - choices: (S,B,T) of 'A'/'B'
      - EV: (S,B,T+1,2)
      - ch_prob: (S,B,T,2)
      - nll: (S,B,T)
      - rewards_self/other/noone: (B,T,2)   # same for all subjects unless customized
      - PE_self/other/noone: (S,B,T)
      - rewards: list of length S with the 3 reward arrays (for EMModel.recover)
      - params: (S,4) = np.array([beta, a_self, a_other, a_noone]).T
    """
    if params.ndim != 2 or params.shape[1] != 4:
        raise ValueError("params must be (nsubjects, 4) = [beta, a_self, a_other, a_noone]")

    S = params.shape[0]
    beta_all     = params[:, 0]
    a_self_all   = params[:, 1]
    a_other_all  = params[:, 2]
    a_noone_all  = params[:, 3]

    if not ((beta_all >= 1e-5) & (beta_all <= 20.0)).all():
        raise ValueError("beta out of bounds [1e-5, 20]")
    if not ((a_self_all >= 0.0) & (a_self_all <= 1.0)).all():
        raise ValueError("a_self must be in [0,1]")
    if not ((a_other_all >= 0.0) & (a_other_all <= 1.0)).all():
        raise ValueError("a_other must be in [0,1]")
    if not ((a_noone_all >= 0.0) & (a_noone_all <= 1.0)).all():
        raise ValueError("a_noone must be in [0,1]")

    rng = np.random.default_rng()
    choice_labels = np.array(['A','B'], dtype=object)

    # outcomes
    if outcomes is not None:
        if outcomes.shape != (nblocks, ntrials, 2, 3):
            raise ValueError("outcomes must be (nblocks, ntrials, 2, 3) with last dim [self, other, noone]")
        Rself  = outcomes[..., 0].astype(float)
        Rother = outcomes[..., 1].astype(float)
        Rnone  = outcomes[..., 2].astype(float)
    else:
        # Bernoulli(0/1) per option/channel
        Rself  = rng.integers(0, 2, size=(nblocks, ntrials, 2)).astype(float)
        Rother = rng.integers(0, 2, size=(nblocks, ntrials, 2)).astype(float)
        Rnone  = rng.integers(0, 2, size=(nblocks, ntrials, 2)).astype(float)

    # alloc
    EV        = np.zeros((S, nblocks, ntrials + 1, 2), dtype=float)
    ch_prob   = np.zeros((S, nblocks, ntrials, 2), dtype=float)
    choices   = np.empty((S, nblocks, ntrials), dtype=object)
    nll       = np.zeros((S, nblocks, ntrials), dtype=float)
    PE_self   = np.zeros((S, nblocks, ntrials), dtype=float)
    PE_other  = np.zeros((S, nblocks, ntrials), dtype=float)
    PE_noone  = np.zeros((S, nblocks, ntrials), dtype=float)

    for s in range(S):
        beta    = float(beta_all[s])
        a_self  = float(a_self_all[s])
        a_other = float(a_other_all[s])
        a_noone = float(a_noone_all[s])

        for b in range(nblocks):
            EV[s, b, 0, :] = 0.0
            for t in range(ntrials):
                p = softmax(EV[s, b, t, :], beta)   # over A/B
                ch_prob[s, b, t, :] = p
                c = int(rng.choice([0, 1], p=p))    # 0='A', 1='B'
                choices[s, b, t] = choice_labels[c]

                v        = EV[s, b, t, c]
                r_self   = float(Rself[b, t, c])
                r_other  = float(Rother[b, t, c])
                r_noone  = float(Rnone[b, t, c])

                pe_self   = r_self  - v
                pe_other  = r_other - v
                pe_noone  = r_noone - v
                PE_self[s,  b, t] = pe_self
                PE_other[s, b, t] = pe_other
                PE_noone[s, b, t] = pe_noone

                delta = a_self * pe_self + a_other * pe_other + a_noone * pe_noone
                EV[s, b, t + 1, :] = EV[s, b, t, :]
                EV[s, b, t + 1, c] = v + delta

                nll[s, b, t] = -np.log(ch_prob[s, b, t, c] + 1e-12)

    # rewards payload (same arrays for all subjects here)
    rewards_payload = [
        {"rewards_self": Rself, "rewards_other": Rother, "rewards_noone": Rnone}
        for _ in range(S)
    ]

    return {
        "params"        : np.column_stack([beta_all, a_self_all, a_other_all, a_noone_all]),
        "choices"       : choices,
        "EV"            : EV,
        "ch_prob"       : ch_prob,
        "nll"           : nll,
        "rewards_self"  : Rself,
        "rewards_other" : Rother,
        "rewards_noone" : Rnone,
        "PE_self"       : PE_self,
        "PE_other"      : PE_other,
        "PE_noone"      : PE_noone,
        "rewards"       : rewards_payload,  # for EMModel.recover(pr_inputs=['choices','rewards'])
    }

# --------------------------------------
# 3α-1β FIT (Lockwood et al., 2016)
# --------------------------------------
def rw3a1b_fit(params: np.ndarray,
               choices: np.ndarray,
               rewards,
               prior=None,
               output: str = "npl"):
    """
    choices: (B,T) of 'A'/'B'
    rewards: dict with keys {'rewards_self','rewards_other','rewards_noone'}
             or tuple (Rself, Rother, Rnone).
    """
    # unpack rewards
    if isinstance(rewards, dict):
        Rself  = rewards["rewards_self"]   # (B,T,2)
        Rother = rewards["rewards_other"]  # (B,T,2)
        Rnone  = rewards["rewards_noone"]  # (B,T,2)
    else:
        Rself, Rother, Rnone = rewards

    nblocks, ntrials = Rself.shape[:2]
    choice_to_idx = {'A': 0, 'B': 1}

    # normalized → natural
    beta    = float(norm2beta(params[0]))
    a_self  = float(norm2alpha(params[1]))
    a_other = float(norm2alpha(params[2]))
    a_noone = float(norm2alpha(params[3]))

    # bounds
    if not (1e-5 <= beta <= 20.0):
        return 1e7
    for a in (a_self, a_other, a_noone):
        if not (0.0 <= a <= 1.0):
            return 1e7

    EV        = np.zeros((nblocks, ntrials + 1, 2), dtype=float)
    ch_prob   = np.zeros((nblocks, ntrials, 2), dtype=float)
    PE_self   = np.zeros((nblocks, ntrials), dtype=float)
    PE_other  = np.zeros((nblocks, ntrials), dtype=float)
    PE_noone  = np.zeros((nblocks, ntrials), dtype=float)
    NLL = 0.0

    for b in range(nblocks):
        EV[b, 0, :] = 0.0
        for t in range(ntrials):
            p = softmax(EV[b, t, :], beta)
            ch_prob[b, t, :] = p

            cchr = choices[b, t]
            c    = choice_to_idx[cchr]
            NLL += -np.log(p[c] + 1e-12)

            v        = EV[b, t, c]
            r_self   = float(Rself[b, t, c])
            r_other  = float(Rother[b, t, c])
            r_noone  = float(Rnone[b, t, c])

            pe_self   = r_self  - v
            pe_other  = r_other - v
            pe_noone  = r_noone - v
            PE_self[b,  t] = pe_self
            PE_other[b, t] = pe_other
            PE_noone[b, t] = pe_noone

            delta = a_self * pe_self + a_other * pe_other + a_noone * pe_noone
            EV[b, t + 1, :] = EV[b, t, :]
            EV[b, t + 1, c] = v + delta

    if output == "all":
        n = nblocks * ntrials
        k = len(params)
        BIC = k * np.log(n) + 2.0 * NLL
        return {
            "params"        : [beta, a_self, a_other, a_noone],
            "EV"            : EV,
            "choices"       : choices,
            "ch_prob"       : ch_prob,
            "rewards_self"  : Rself,
            "rewards_other" : Rother,
            "rewards_noone" : Rnone,
            "PE_self"       : PE_self,
            "PE_other"      : PE_other,
            "PE_noone"      : PE_noone,
            "nll"           : NLL,
            "BIC"           : BIC,
        }

    return calc_fval(NLL, params, prior=prior, output=output)

# ----------------------------------------
# 1Q–4α–1β  SIMULATE (Rhoads et al., 2025)
# ----------------------------------------
def gen_rnd_blocks(items, nblocks=2, nsubjects=100):
    perms = list(permutations(items))
    for _ in range(nsubjects):
        # Randomly pick nblocks permutations with replacement
        blocks = random.choices(perms, k=nblocks)
        combined = tuple(chain.from_iterable(blocks))
        yield combined

def rw4a1b_sim(params: np.ndarray,
                    nblocks: int = 12,
                    ntrials: int = 20):
    """
    Simulate a 4-option 1Q RW with one beta and four learning rates:
      a_self_pos, a_self_neg, a_other_pos, a_other_neg

    Each trial shows a PAIR OF OPTIONS (indices 0..3) and the agent picks one of those two.
    Outcomes for SELF and OTHER are drawn independently from option-specific marginals over {-1,0,+1}

    Fixed design:
      - There are 6 unique option-pairs from {0,1,2,3}: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3).
      - For 12 blocks, we cycle those 6 pairs twice (blocks 0..11).
      - Each block also has a fixed pattern type for outcome marginals, cycled over 4 types:
            (+/+), (+/-), (-/+), (-/-)  and then repeat.
        Pattern type is fixed within a block.

    Returns:
      - choices          : (S,B,T) int indices 0..3 (chosen option among the shown pair)
      - outcomes_self    : (S,B,T) int  in {-1,0,+1}
      - outcomes_other   : (S,B,T) int  in {-1,0,+1}
      - option_pairs     : (S,B,T,2) int indices for the two shown options on each trial
      - also EV, ch_prob (over 4), and PE components
    """
    assert nblocks % 6 == 0, "nblocks should be multiple of 6 for full counterbalancing"
    
    # Bounds check
    beta_all, a_self_pos_all, a_self_neg_all, a_other_pos_all, a_other_neg_all = (params[:, i].astype(float) for i in range(5))
    if not ((beta_all > 1e-5) & (beta_all <= 20.0)).all():
        raise ValueError("beta out of bounds")
    for arr, name in [(a_self_pos_all, "a_self_pos"), (a_self_neg_all, "a_self_neg"),
                      (a_other_pos_all, "a_other_pos"), (a_other_neg_all, "a_other_neg")]:
        if not ((0.0 <= arr) & (arr <= 1.0)).all():
            raise ValueError(f"{name} out of bounds")

    rng = np.random.default_rng()
    nsubjects = params.shape[0]

    # Outputs 
    choices        = np.zeros((nsubjects, nblocks, ntrials), dtype=object) # A,B,C,D
    outcomes_self  = np.zeros((nsubjects, nblocks, ntrials), dtype=int)    # -1/0/+1
    outcomes_other = np.zeros((nsubjects, nblocks, ntrials), dtype=int)    # -1/0/+1
    option_pairs   = np.zeros((nsubjects, nblocks, ntrials, 2), dtype=object)

    # Optional diagnostics
    EV           = np.zeros((nsubjects, nblocks, ntrials + 1, 4), dtype=float)
    ch_prob      = np.zeros((nsubjects, nblocks, ntrials, 4), dtype=float)
    pe_self      = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    pe_other     = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    pe_self_pos  = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    pe_self_neg  = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    pe_other_pos = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    pe_other_neg = np.zeros((nsubjects, nblocks, ntrials), dtype=float)

    # create task structure
    all_pairs = ['AB', 'AC', 'AD', 'BC', 'BD', 'CD']
    letter_to_idx = {'A':0, 'B':1, 'C':2, 'D':3}
    block_orders = list(gen_rnd_blocks(['AB', 'AC', 'AD', 'BC', 'BD', 'CD'], 
                        nblocks=nblocks, nsubjects=nsubjects))
    if ntrials == 20:
        # high 75%, mid 15%, low 10%
        opt_templates = {'+': [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 0, 0, 0, -1, -1],
                         '-': [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0,  1,  1]}
    else: # get proportion of good (1,0,-1) and bad (-1,0,1) at high 75%, mid 15%, low 10%
        opt_templates = {'+': np.random.choice([1, 0, -1], size=ntrials, p=[0.75, 0.15, 0.10]),
                         '-': np.random.choice([-1, 0, 1], size=ntrials, p=[0.75, 0.15, 0.10])}
    opt_types = {'A': ('+','+'), 'B': ('+','-'),'C': ('-','+'), 'D': ('-','-')}

    for s in range(nsubjects):
        beta        = beta_all[s]
        a_self_pos  = a_self_pos_all[s]
        a_self_neg  = a_self_neg_all[s]
        a_other_pos = a_other_pos_all[s]
        a_other_neg = a_other_neg_all[s]

        for b in range(nblocks):
            # get block pair
            opt1, opt2 = block_orders[s][b]
            o1, o2 = letter_to_idx[opt1], letter_to_idx[opt2]

            # create possible outcomes for this block
            all_outcomes = {'A_self' :[np.nan]*ntrials, 'A_other':[np.nan]*ntrials,
                            'B_self' :[np.nan]*ntrials, 'B_other':[np.nan]*ntrials,
                            'C_self' :[np.nan]*ntrials, 'C_other':[np.nan]*ntrials,
                            'D_self' :[np.nan]*ntrials, 'D_other':[np.nan]*ntrials
                            }
            for this_opt in (opt1, opt2):
                self_kind, other_kind = opt_types[this_opt]
                all_outcomes[f'{this_opt}_self']  = rng.permutation(opt_templates[self_kind])
                all_outcomes[f'{this_opt}_other'] = rng.permutation(opt_templates[other_kind])
            
            EV[s, b, 0, :] = 0
            for t in range(ntrials):
                # the two shown options on this trial (fixed per block)
                option_pairs[s, b, t, 0] = opt1
                option_pairs[s, b, t, 1] = opt2
                
                # softmax over the two shown options
                shown_vals = np.array([EV[s, b, t, o1], EV[s, b, t, o2]], dtype=float)
                p = softmax(shown_vals, beta)
                ch_prob[s, b, t, o1] = p[0]
                ch_prob[s, b, t, o2] = p[1]
                choices[s, b, t] = rng.choice([opt1, opt2], p=p)
                c = letter_to_idx[choices[s, b, t]]

                # get outcomes from choices and all_outcomes
                outcomes_self[s, b, t] = all_outcomes[f'{choices[s, b, t]}_self'][t]
                outcomes_other[s, b, t] = all_outcomes[f'{choices[s, b, t]}_other'][t]

                # compute prediction errors
                pe_self[s, b, t] = outcomes_self[s, b, t] - EV[s, b, t, c]
                pe_other[s, b, t] = outcomes_other[s, b, t] - EV[s, b, t, c]

                pe_self_pos[s, b, t]  = pe_self[s, b, t] if pe_self[s, b, t]   >= 0.0 else 0.0
                pe_self_neg[s, b, t]  = pe_self[s, b, t] if pe_self[s, b, t]   <  0.0 else 0.0
                pe_other_pos[s, b, t] = pe_other[s, b, t] if pe_other[s, b, t] >= 0.0 else 0.0
                pe_other_neg[s, b, t] = pe_other[s, b, t] if pe_other[s, b, t] <  0.0 else 0.0

                # update the chosen option
                EV[s, b, t+1, :] = EV[s, b, t, :].copy()
                EV[s, b, t+1, c] = EV[s, b, t, c] + (a_self_pos  * pe_self_pos[s, b, t] + 
                                                     a_self_neg  * pe_self_neg[s, b, t] + 
                                                     a_other_pos * pe_other_pos[s, b, t] + 
                                                     a_other_neg * pe_other_neg[s, b, t])

    return {"params": params,
            "choices": choices,                # chosen option indices (A,B,C,D)
            "outcomes_self": outcomes_self,    # -1/0/+1
            "outcomes_other": outcomes_other,  # -1/0/+1
            "option_pairs": option_pairs,      # which two options were shown on each trial
            "EV": EV,
            "ch_prob": ch_prob,
            "pe_self": pe_self,
            "pe_other": pe_other,
            "pe_self_pos": pe_self_pos,
            "pe_self_neg": pe_self_neg,
            "pe_other_pos": pe_other_pos,
            "pe_other_neg": pe_other_neg,
            }

# ----------------------------------
# 1Q–4α–1β FIT (Rhoads et al., 2025)
# ----------------------------------
def rw4a1b_fit(params: np.ndarray,
               choices: np.ndarray,        # (B,T) chosen options (A,B,C,D)
               outcomes_self: np.ndarray,  # (B,T) in {-1,0,+1}
               outcomes_other: np.ndarray, # (B,T) in {-1,0,+1}
               option_pairs: np.ndarray,   # (B,T,2) indices of shown options per trial
               prior=None,
               output: str = "npl"):

    beta        = norm2beta(params[0])
    a_self_pos  = norm2alpha(params[1])
    a_self_neg  = norm2alpha(params[2])
    a_other_pos = norm2alpha(params[3])
    a_other_neg = norm2alpha(params[4])

    # Bounds
    if not (1e-5 <= beta <= 20.0):
        return 1e7
    for a in (a_self_pos, a_self_neg, a_other_pos, a_other_neg):
        if not (0.0 <= a <= 1.0):
            return 1e7

    # Convert choices (accepts letters or indices)
    choices_arr = np.asarray(choices)
    if not np.issubdtype(choices_arr.dtype, np.number):
        letter_to_idx = {'A':0, 'B':1, 'C':2, 'D':3}
        choices_arr = np.vectorize(letter_to_idx.get)(choices_arr)
    choices_arr = choices_arr.astype(int, copy=False)

    nblocks, ntrials = outcomes_self.shape
    EV           = np.zeros((nblocks, ntrials + 1, 4), dtype=float)
    ch_prob      = np.zeros((nblocks, ntrials, 4), dtype=float)
    pe_self      = np.zeros((nblocks, ntrials), dtype=float)
    pe_other     = np.zeros((nblocks, ntrials), dtype=float)
    pe_self_pos  = np.zeros((nblocks, ntrials), dtype=float)
    pe_self_neg  = np.zeros((nblocks, ntrials), dtype=float)
    pe_other_pos = np.zeros((nblocks, ntrials), dtype=float)
    pe_other_neg = np.zeros((nblocks, ntrials), dtype=float)

    nll = 0.0
    for b in range(nblocks):
        EV[b, 0, :] = 0.0
        for t in range(ntrials):
            # get shown options
            opt1, opt2 = option_pairs[b, t, 0], option_pairs[b, t, 1]
            o1, o2 = letter_to_idx[opt1], letter_to_idx[opt2]

            # get probability of the chosen option
            c = letter_to_idx[choices[b, t]]
            shown_vals = np.array([EV[b, t, o1], EV[b, t, o2]], dtype=float)
            probs_two  = softmax(shown_vals, beta)  # len=2
            ch_prob[b, t, o1] = probs_two[0]
            ch_prob[b, t, o2] = probs_two[1]
            nll += -np.log(probs_two[0] if c == o1 else probs_two[1] + 1e-12)

            # compute prediction errors
            pe_self[b, t]  = outcomes_self[b, t]  - EV[b, t, c]
            pe_other[b, t] = outcomes_other[b, t] - EV[b, t, c]

            pe_self_pos[b, t]  = pe_self[b, t]  if pe_self[b, t]  >= 0.0 else 0.0
            pe_self_neg[b, t]  = pe_self[b, t]  if pe_self[b, t]  <  0.0 else 0.0
            pe_other_pos[b, t] = pe_other[b, t] if pe_other[b, t] >= 0.0 else 0.0
            pe_other_neg[b, t] = pe_other[b, t] if pe_other[b, t] <  0.0 else 0.0

            # update chosen option
            EV[b, t+1, :] = EV[b, t, :].copy()
            EV[b, t+1, c] = EV[b, t, c] + (a_self_pos  * pe_self_pos[b, t] +
                                           a_self_neg  * pe_self_neg[b, t] +
                                           a_other_pos * pe_other_pos[b, t] +
                                           a_other_neg * pe_other_neg[b, t])

    if output == "all":
        return {
            "params": [beta, a_self_pos, a_self_neg, a_other_pos, a_other_neg],
            "choices": choices_arr,
            "outcomes_self": outcomes_self,
            "outcomes_other": outcomes_other,
            "option_pairs": option_pairs,
            "EV": EV,
            "nll": nll,
            "ch_prob": ch_prob,
            "pe_self": pe_self,
            "pe_other": pe_other,
            "pe_self_pos": pe_self_pos,
            "pe_self_neg": pe_self_neg,
            "pe_other_pos": pe_other_pos,
            "pe_other_neg": pe_other_neg,
        }
    else:
        return calc_fval(nll, params, prior=prior, output=output)