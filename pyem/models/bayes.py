import numpy as np
from ..utils.math import norm2alpha, calc_fval

def _generate_fishp(lambda1: float, n_fish: int) -> np.ndarray:
    """Return transition matrix for observing fish colours.

    ``lambda1`` is the probability of seeing the same colour again. The
    remaining probability mass is split equally among the other colours. Small
    ``lambda1`` values imply slower belief updates because each observation
    carries less weight.
    """
    m = lambda1
    s = (1 - lambda1) / (n_fish - 1)
    fishp = np.eye(n_fish) * m + (1 - np.eye(n_fish)) * s
    return fishp

def bayes_sim(params: np.ndarray, nblocks: int = 10, ntrials: int = 15,
             n_fish: int = 3) -> dict:
    """Simulate the fish task described in the repository documentation.

    A new coloured fish appears on every trial. Participants guess which pond
    (out of three) the fish came from without receiving feedback. The
    parameter ``lambda1`` controls how strongly an observation depends on the
    previous one; smaller values slow belief updates, requiring more
    confirming evidence to increase confidence.
    """
    n_subjects = params.shape[0]
    # Pond compositions: each row gives colour probabilities for a pond
    pond_distributions = np.array([[0.8, 0.1, 0.1],
                                   [0.1, 0.8, 0.1],
                                   [0.1, 0.1, 0.8]])

    choices = np.empty((n_subjects, nblocks, ntrials), dtype=int)
    observations = np.empty((n_subjects, nblocks, ntrials), dtype=int)
    probabilities = np.empty((n_subjects, nblocks, ntrials + 1, n_fish))
    ponds = np.empty((n_subjects, nblocks, ntrials), dtype=int)
    rng = np.random.default_rng()

    for s in range(n_subjects):
        lambda1 = params[s, 0]
        fishp = _generate_fishp(lambda1, n_fish)
        # determine which pond is correct for each block
        base = np.array([0, 1, 2] * (nblocks // 3 + 1))
        block_to_pond = rng.permutation(base[:nblocks])
        for b in range(nblocks):
            pond_type = block_to_pond[b]
            # sequence of observed fish colours for this block
            fish_disp = rng.choice(n_fish, size=ntrials,
                                   p=pond_distributions[pond_type])
            pondp = np.ones((ntrials + 1, n_fish)) / n_fish  # prior over ponds
            probabilities[s, b, 0, :] = pondp[0, :]
            for t in range(ntrials):
                ponds[s, b, t] = pond_type
                observations[s, b, t] = fish_disp[t]
                den = np.sum(pondp[t, :] * fishp[fish_disp[t], :])
                pondp[t + 1, :] = (fishp[fish_disp[t], :] * pondp[t, :]) / den
                pondp[t + 1, :] /= np.sum(pondp[t + 1, :])
                probabilities[s, b, t + 1, :] = pondp[t + 1, :]
                # choice corresponds to sampled pond based on posterior
                choice = rng.choice(n_fish, p=pondp[t + 1, :])
                choices[s, b, t] = choice
    return {
        "params": params,
        "choices": choices,
        "observations": observations,
        "probabilities": probabilities,
        "ponds": ponds,
    }

def bayes_fit(params, choices, observations, prior=None, output: str = 'npl'):
    """Likelihood for the fish task.

    Parameters are supplied in Gaussian space and transformed to ``lambda1``
    (0--1) using :func:`norm2alpha`. ``lambda1`` governs the pace of belief
    updating; smaller values require more confirming evidence to reduce
    uncertainty. The function can return either the negative log-likelihood
    (``output='nll'``), the negative posterior likelihood (``'npl'``) or, with
    ``'all'``, additional diagnostic values.
    """
    lambda1 = norm2alpha(params[0])
    nblocks, ntrials = choices.shape
    n_fish = 3
    fishp = _generate_fishp(lambda1, n_fish)
    nll = 0.0
    for b in range(nblocks):
        pondp = np.ones((ntrials + 1, n_fish)) / n_fish
        for t in range(ntrials):
            fish_disp = observations[b, t]
            real_choice = choices[b, t]
            den = np.sum(pondp[t, :] * fishp[fish_disp, :])
            pondp[t + 1, :] = (fishp[fish_disp, :] * pondp[t, :]) / den
            pondp[t + 1, :] /= np.sum(pondp[t + 1, :])
            nll += -np.log(pondp[t + 1, real_choice] + 1e-12)
    if output == 'all':
        return {"params": np.array([lambda1]), "nll": nll}

    # return requested objective value
    return calc_fval(nll, params, prior=prior, output=output)
