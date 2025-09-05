
from __future__ import annotations
import numpy as np
from ..utils.math import norm2alpha
from ..utils.stats import calc_BICint

def _generate_fishp(lambda1: float, n_fish: int):
    m = lambda1
    s = (1 - lambda1) / (n_fish - 1)
    fishp = np.eye(n_fish) * m + (1 - np.eye(n_fish)) * s
    return fishp

def simulate(params: np.ndarray, n_blocks=10, n_trials=15, n_fish=3):
    n_subjects = params.shape[0]
    pond_distributions = np.array([[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8]])
    choices = np.empty((n_subjects, n_blocks, n_trials), dtype=int)
    observations = np.empty((n_subjects, n_blocks, n_trials), dtype=int)
    probabilities = np.empty((n_subjects, n_blocks, n_trials+1, n_fish))
    ponds = np.empty((n_subjects, n_blocks, n_trials), dtype=int)
    rng = np.random.default_rng()
    for s in range(n_subjects):
        lambda1 = params[s, 0]
        fishp = _generate_fishp(lambda1, n_fish)
        base = np.array([0,1,2] * (n_blocks//3 + 1))
        block_to_pond = rng.permutation(base[:n_blocks])
        for b in range(n_blocks):
            pond_type = block_to_pond[b]
            fish_disp = rng.choice(n_fish, size=n_trials, p=pond_distributions[pond_type])
            pondp = np.ones((n_trials+1, n_fish)) / n_fish
            probabilities[s, b, 0, :] = pondp[0, :]
            for t in range(n_trials):
                ponds[s, b, t] = pond_type
                observations[s, b, t] = fish_disp[t]
                den = np.sum(pondp[t, :] * fishp[fish_disp[t], :])
                pondp[t+1, :] = (fishp[fish_disp[t], :] * pondp[t, :]) / den
                pondp[t+1, :] /= np.sum(pondp[t+1, :])
                probabilities[s, b, t+1, :] = pondp[t+1, :]
                choice = rng.choice(n_fish, p=pondp[t+1, :])
                choices[s, b, t] = choice
    return {"params": params, "choices": choices, "observations": observations, "probabilities": probabilities, "ponds": ponds}

def fit(params, choices, observations, prior=None, output='npl'):
    lambda1 = norm2alpha(params[0])
    n_blocks, n_trials = choices.shape
    n_fish = 3
    fishp = _generate_fishp(lambda1, n_fish)
    NLL = 0.0
    for b in range(n_blocks):
        pondp = np.ones((n_trials+1, n_fish)) / n_fish
        for t in range(n_trials):
            fish_disp = observations[b, t]
            real_choice = choices[b, t]
            den = np.sum(pondp[t, :] * fishp[fish_disp, :])
            pondp[t+1, :] = (fishp[fish_disp, :] * pondp[t, :]) / den
            pondp[t+1, :] /= np.sum(pondp[t+1, :])
            NLL += -np.log(pondp[t+1, real_choice] + 1e-12)
    if output == 'all':
        return {"params": np.array([lambda1]), "NLL": NLL}
    # map vs mle return
    if prior is not None and output == 'npl' and hasattr(prior, 'logpdf'):
        return NLL + (-prior.logpdf(np.asarray(params)))
    return NLL
