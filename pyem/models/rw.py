
from __future__ import annotations
import numpy as np
from ..utils.math import softmax, norm2alpha, norm2beta

def rw_simulate(params: np.ndarray, nblocks: int = 3, ntrials: int = 24, outcomes: np.ndarray | None = None):
    """
    Vectorized simulation of a two-armed bandit under a Rescorla-Wagner model.
    params: (nsubjects, 2) in natural space: [beta_norm, lr_norm]
            beta = norm2beta(beta_norm, max_val=20)
            lr   = norm2alpha(lr_norm)
    """
    nsubjects = params.shape[0]
    choices = np.empty((nsubjects, nblocks, ntrials), dtype=object)
    rewards = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    EV = np.zeros((nsubjects, nblocks, ntrials + 1, 2), dtype=float)
    CH_PROB = np.zeros((nsubjects, nblocks, ntrials, 2), dtype=float)
    CHOICES_A = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    PE = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    CHOICE_NLL = np.zeros((nsubjects, nblocks, ntrials), dtype=float)

    rng = np.random.default_rng()
    this_block_probs = np.array([0.8, 0.2])

    for s in range(nsubjects):
        beta = float(norm2beta(params[s, 0]))
        lr = float(norm2alpha(params[s, 1]))
        for b in range(nblocks):
            EV[s, b, 0, :] = 0.5
            for t in range(ntrials):
                p = softmax(EV[s, b, t, :], beta)
                CH_PROB[s, b, t, :] = p
                c = rng.choice([0, 1], p=p)
                choices[s, b, t] = "A" if c == 0 else "B"
                CHOICES_A[s, b, t] = 1.0 if c == 0 else 0.0
                if outcomes is None:
                    rew = rng.choice([1.0, 0.0], p=this_block_probs if c == 0 else this_block_probs[::-1])
                else:
                    rew = float(outcomes[b, t, c])
                rewards[s, b, t] = rew
                PE[s, b, t] = rew - EV[s, b, t, c]
                EV[s, b, t + 1, :] = EV[s, b, t, :]
                EV[s, b, t + 1, c] = EV[s, b, t, c] + lr * PE[s, b, t]
                CHOICE_NLL[s, b, t] = -np.log(p[c] + 1e-12)

    return {
        "params": params,
        "choices": choices,
        "rewards": rewards,
        "EV": EV,
        "CH_PROB": CH_PROB,
        "CHOICES_A": CHOICES_A,
        "PE": PE,
        "CHOICE_NLL": CHOICE_NLL,
    }

def rw_fit(params, choices, rewards, prior=None, output="npl"):
    """
    A thin adapter compatible with EM: returns NPL or NLL.
    params: (2,) in normalized space
    """
    beta = float(norm2beta(params[0]))
    lr = float(norm2alpha(params[1]))

    # bounds checks
    if not (1e-5 <= beta <= 20.0): return 1e7
    if not (0.0 <= lr <= 1.0): return 1e7

    nblocks, ntrials = rewards.shape
    EV = np.zeros((nblocks, ntrials + 1, 2))
    CHOICE_NLL = 0.0
    for b in range(nblocks):
        EV[b, 0, :] = 0.5
        for t in range(ntrials):
            c = 0 if choices[b, t] == "A" else 1
            p = softmax(EV[b, t, :], beta)
            r = rewards[b, t]
            pe = r - EV[b, t, c]
            EV[b, t + 1, :] = EV[b, t, :]
            EV[b, t + 1, c] = EV[b, t, c] + lr * pe
            CHOICE_NLL += -np.log(p[c] + 1e-12)

    if output == "nll":
        return CHOICE_NLL

    # negative posterior likelihood
    if prior is not None:
        nlp = -prior.logpdf(np.asarray(params))
        return CHOICE_NLL + nlp
    else:
        # if no prior, interpret as nll (legacy safety)
        return CHOICE_NLL
