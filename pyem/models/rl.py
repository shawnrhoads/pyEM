
from __future__ import annotations
import numpy as np
from ..utils.math import softmax, norm2alpha, norm2beta, calc_fval

def rw1a1b_simulate(params: np.ndarray, nblocks: int = 3, ntrials: int = 24,
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


def rw2a1b_simulate(params: np.ndarray, nblocks: int = 3, ntrials: int = 24,
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

# --------------------------
# 3α-1β SIMULATE (natural)
# --------------------------
def rw3a1b_simulate(params: np.ndarray,
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


# ----------------------
# 3α-1β FIT (normalized)
# ----------------------
def rw3a1b_fit(params: np.ndarray,
               choices: np.ndarray,
               rewards,
               prior=None,
               output: str = "npl"):
    """
    NORMALIZED params -> natural via norm2beta/norm2alpha.

    rewards: dict with keys {'rewards_self','rewards_other','rewards_noone'}
             or tuple (Rself, Rother, Rnone).
    choices: (B,T) of 'A'/'B'
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


# ---------------------------
# 1Q-4α-1β SIMULATE (natural)
# ---------------------------
def rw4a1b_simulate(params: np.ndarray,
                    nblocks: int = 12,
                    ntrials: int = 20,
                    outcomes: np.ndarray | None = None):
    """
    4-option RW with one EV vector, one beta, and four learning rates:
      a_self_pos, a_self_neg, a_other_pos, a_other_neg  (NATURAL space).

    params:  (S,5) = [beta, a_self_pos, a_self_neg, a_other_pos, a_other_neg]
    outcomes (optional): (B, T, 4, 2) with last dim [self, other]; values can be real.
                         If None, simulate ternary {-1, 0, +1} (zeros included).

    Returns dict with keys:
      - params (S,5)
      - choices (S,B,T) of 'A'/'B'/'C'/'D'
      - EV (S,B,T+1,4)
      - ch_prob (S,B,T,4)
      - nll (S,B,T)
      - rewards_self (B,T,4), rewards_other (B,T,4)
      - option_pairs (B,T) of 'AB','AC',...
      - PE_self, PE_other (S,B,T)  [signed]
      - PE_self_pos, PE_self_neg, PE_other_pos, PE_other_neg (S,B,T)  [split]
      - rewards: list of length S, each a dict {'rewards_self','rewards_other','option_pairs'} for EMModel.recover
    """
    if params.ndim != 2 or params.shape[1] != 5:
        raise ValueError("params must be (nsubjects, 5) = [beta, a_self_pos, a_self_neg, a_other_pos, a_other_neg]")

    S = params.shape[0]
    beta_all  = params[:, 0]
    alpha_all = params[:, 1:5]

    if not ((beta_all >= 1e-5) & (beta_all <= 20.0)).all():
        raise ValueError("beta out of bounds [1e-5, 20]")
    if not ((alpha_all >= 0.0) & (alpha_all <= 1.0)).all():
        raise ValueError("alphas must be in [0,1]")

    rng = np.random.default_rng()

    # pair mapping
    option_map = {
        'AB': [0, 1], 'BA': [0, 1],
        'AC': [0, 2], 'CA': [0, 2],
        'AD': [0, 3], 'DA': [0, 3],
        'BC': [1, 2], 'CB': [1, 2],
        'BD': [1, 3], 'DB': [1, 3],
        'CD': [2, 3], 'DC': [2, 3],
    }
    labels = np.array(['A','B','C','D'], dtype=object)

    # option pairs per trial
    keys = list(option_map.keys())
    option_pairs = np.array([[rng.choice(keys) for _ in range(ntrials)]
                             for _ in range(nblocks)], dtype=object)

    # outcomes (B,T,4,2) → self/other (B,T,4)
    if outcomes is not None:
        if outcomes.shape != (nblocks, ntrials, 4, 2):
            raise ValueError("outcomes must be (nblocks, ntrials, 4, 2) with last dim [self, other]")
        rewards_self  = outcomes[..., 0].astype(float)
        rewards_other = outcomes[..., 1].astype(float)
    else:
        # TERNARY {-1, 0, +1} with zeros included (tweak probs as needed)
        vals = np.array([-1.0, 0.0, 1.0])
        p_self  = np.array([0.25, 0.50, 0.25])  # P(-1), P(0), P(+1)
        p_other = np.array([0.30, 0.40, 0.30])
        rewards_self  = rng.choice(vals, size=(nblocks, ntrials, 4), p=p_self).astype(float)
        rewards_other = rng.choice(vals, size=(nblocks, ntrials, 4), p=p_other).astype(float)

    # allocate
    EV           = np.zeros((S, nblocks, ntrials + 1, 4), dtype=float)
    ch_prob      = np.zeros((S, nblocks, ntrials, 4), dtype=float)
    choices      = np.empty((S, nblocks, ntrials), dtype=object)
    nll          = np.zeros((S, nblocks, ntrials), dtype=float)

    PE_self      = np.zeros((S, nblocks, ntrials), dtype=float)
    PE_other     = np.zeros((S, nblocks, ntrials), dtype=float)
    PE_self_pos  = np.zeros((S, nblocks, ntrials), dtype=float)
    PE_self_neg  = np.zeros((S, nblocks, ntrials), dtype=float)
    PE_other_pos = np.zeros((S, nblocks, ntrials), dtype=float)
    PE_other_neg = np.zeros((S, nblocks, ntrials), dtype=float)

    for s in range(S):
        beta, a_self_pos, a_self_neg, a_other_pos, a_other_neg = map(float, params[s, :])
        for b in range(nblocks):
            EV[s, b, 0, :] = 0.0
            for t in range(ntrials):
                pair = option_pairs[b, t]
                opts = option_map[pair]
                p    = softmax(EV[s, b, t, opts], beta)
                ch_prob[s, b, t, opts] = p
                c    = int(rng.choice(opts, p=p))
                choices[s, b, t] = labels[c]

                v   = EV[s, b, t, c]
                r_self  = float(rewards_self[b, t, c])
                r_other = float(rewards_other[b, t, c])

                pe_self  = r_self  - v
                pe_other = r_other - v
                PE_self[s,  b, t] = pe_self
                PE_other[s, b, t] = pe_other

                # split components
                PE_self_pos[s,  b, t]  = max(pe_self,  0.0)
                PE_self_neg[s,  b, t]  = min(pe_self,  0.0)
                PE_other_pos[s, b, t]  = max(pe_other, 0.0)
                PE_other_neg[s, b, t]  = min(pe_other, 0.0)

                delta = (a_self_pos  * PE_self_pos[s, b, t]  +
                         a_self_neg  * PE_self_neg[s, b, t]  +
                         a_other_pos * PE_other_pos[s, b, t] +
                         a_other_neg * PE_other_neg[s, b, t])

                EV[s, b, t + 1, :] = EV[s, b, t, :]
                EV[s, b, t + 1, c] = v + delta

                nll[s, b, t] = -np.log(ch_prob[s, b, t, c] + 1e-12)

    # for EMModel.recover(pr_inputs=['choices','rewards'])
    rewards_payload = [
        {"rewards_self": rewards_self, "rewards_other": rewards_other, "option_pairs": option_pairs}
        for _ in range(S)
    ]

    return {
        "params"        : params,
        "choices"       : choices,
        "EV"            : EV,
        "ch_prob"       : ch_prob,
        "nll"           : nll,
        "rewards_self"  : rewards_self,
        "rewards_other" : rewards_other,
        "option_pairs"  : option_pairs,
        "PE_self"       : PE_self,
        "PE_other"      : PE_other,
        "PE_self_pos"   : PE_self_pos,
        "PE_self_neg"   : PE_self_neg,
        "PE_other_pos"  : PE_other_pos,
        "PE_other_neg"  : PE_other_neg,
        "rewards"       : rewards_payload,
    }


# -------------------------
# 1Q-4α-1β FIT (normalized)
# -------------------------
def rw4a1b_fit(params: np.ndarray,
               choices: np.ndarray,
               rewards,
               prior=None,
               output: str = "npl"):
    """
    NORMALIZED params → natural via norm2beta/norm2alpha.

    rewards:
      - dict {'rewards_self','rewards_other','option_pairs'}, or
      - tuple (rewards_self, rewards_other, option_pairs)

    choices: (B,T) of 'A'/'B'/'C'/'D'
    """
    # unpack rewards
    if isinstance(rewards, dict):
        Rself  = rewards["rewards_self"]   # (B,T,4)
        Rother = rewards["rewards_other"]  # (B,T,4)
        pairs  = rewards["option_pairs"]   # (B,T)
    else:
        Rself, Rother, pairs = rewards

    nblocks, ntrials = Rself.shape[:2]

    option_map = {
        'AB': [0, 1], 'BA': [0, 1],
        'AC': [0, 2], 'CA': [0, 2],
        'AD': [0, 3], 'DA': [0, 3],
        'BC': [1, 2], 'CB': [1, 2],
        'BD': [1, 3], 'DB': [1, 3],
        'CD': [2, 3], 'DC': [2, 3],
    }
    choice_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    # normalized → natural param names
    beta        = float(norm2beta(params[0]))
    a_self_pos  = float(norm2alpha(params[1]))
    a_self_neg  = float(norm2alpha(params[2]))
    a_other_pos = float(norm2alpha(params[3]))
    a_other_neg = float(norm2alpha(params[4]))

    # bounds
    if not (1e-5 <= beta <= 20.0):
        return 1e7
    for a in (a_self_pos, a_self_neg, a_other_pos, a_other_neg):
        if not (0.0 <= a <= 1.0):
            return 1e7

    EV           = np.zeros((nblocks, ntrials + 1, 4), dtype=float)
    ch_prob      = np.zeros((nblocks, ntrials, 4), dtype=float)
    PE_self      = np.zeros((nblocks, ntrials), dtype=float)
    PE_other     = np.zeros((nblocks, ntrials), dtype=float)
    PE_self_pos  = np.zeros((nblocks, ntrials), dtype=float)
    PE_self_neg  = np.zeros((nblocks, ntrials), dtype=float)
    PE_other_pos = np.zeros((nblocks, ntrials), dtype=float)
    PE_other_neg = np.zeros((nblocks, ntrials), dtype=float)

    NLL = 0.0

    for b in range(nblocks):
        EV[b, 0, :] = 0.0
        for t in range(ntrials):
            opts = option_map[pairs[b, t]]
            p    = softmax(EV[b, t, opts], beta)
            ch_prob[b, t, opts] = p

            cchr = choices[b, t]
            c    = choice_to_idx[cchr]
            NLL += -np.log(p[opts.index(c)] + 1e-12)

            v      = EV[b, t, c]
            r_self = float(Rself[b, t, c])
            r_oth  = float(Rother[b, t, c])

            pe_self  = r_self - v
            pe_other = r_oth  - v
            PE_self[b,  t] = pe_self
            PE_other[b, t] = pe_other

            # split components
            PE_self_pos[b,  t] = max(pe_self,  0.0)
            PE_self_neg[b,  t] = min(pe_self,  0.0)
            PE_other_pos[b, t] = max(pe_other, 0.0)
            PE_other_neg[b, t] = min(pe_other, 0.0)

            delta = (a_self_pos  * PE_self_pos[b, t]  +
                     a_self_neg  * PE_self_neg[b, t]  +
                     a_other_pos * PE_other_pos[b, t] +
                     a_other_neg * PE_other_neg[b, t])

            EV[b, t + 1, :] = EV[b, t, :]
            EV[b, t + 1, c] = v + delta

    if output == "all":
        n = nblocks * ntrials
        k = len(params)
        BIC = k * np.log(n) + 2.0 * NLL
        return {
            "params"       : [beta, a_self_pos, a_self_neg, a_other_pos, a_other_neg],
            "EV"           : EV,
            "choices"      : choices,
            "ch_prob"      : ch_prob,
            "rewards_self" : Rself,
            "rewards_other": Rother,
            "option_pairs" : pairs,
            "PE_self"      : PE_self,
            "PE_other"     : PE_other,
            "PE_self_pos"  : PE_self_pos,
            "PE_self_neg"  : PE_self_neg,
            "PE_other_pos" : PE_other_pos,
            "PE_other_neg" : PE_other_neg,
            "nll"          : NLL,
            "BIC"          : BIC,
        }

    return calc_fval(NLL, params, prior=prior, output=output)
