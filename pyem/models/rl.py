
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

    # transform all params
    all_beta = norm2beta(params[:, 0])
    all_alpha = norm2alpha(params[:, 1])

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

    # transform all params
    all_beta = norm2beta(params[:, 0])
    all_alpha_pos = norm2alpha(params[:, 1])
    all_alpha_neg = norm2alpha(params[:, 2])

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
