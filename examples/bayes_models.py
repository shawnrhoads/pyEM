import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from pyEM.math import calc_fval, norm2alpha

def generate_fishp(lambda1, n_fish):
    """
    Generates the transition probability matrix for fish types based on lambda1.
    
    Args:
        lambda1 (float): The parameter defining likelihood distribution.
        n_fish (int): Number of fish types.

    Returns:
        np.ndarray: Transition probability matrix of shape (n_fish, n_fish).
    """
    m = lambda1
    s = (1 - lambda1) / (n_fish - 1)
    fishp = np.eye(n_fish) * m + (1 - np.eye(n_fish)) * s
    return fishp

def simulate(params, n_blocks=10, n_trials=15, n_fish=3):
    """
    Simulates choices and observations for each agent across blocks and trials.

    Args:
        params (array): Array of shape (n_subjects, 1) containing lambda1 for each agent.
        n_blocks (int): Number of blocks per agent (default is 10).
        n_trials (int): Number of trials per block (default is 15).
        n_fish (int): Number of fish types (default is 3).

    Returns:
        dict: Contains simulation results including choices, probabilities, and observations.
    """
    n_subjects = params.shape[0]

    # Define pond distributions for fish displays
    pond_distributions = np.array([
        [0.8, 0.1, 0.1],  # Pond 0 favors fish type 0
        [0.1, 0.8, 0.1],  # Pond 1 favors fish type 1
        [0.1, 0.1, 0.8]   # Pond 2 favors fish type 2
    ])

    choices = np.empty((n_subjects, n_blocks, n_trials), dtype=int)
    observations = np.empty((n_subjects, n_blocks, n_trials), dtype=int)
    probabilities = np.empty((n_subjects, n_blocks, n_trials+1, n_fish))
    ponds = np.empty((n_subjects, n_blocks, n_trials), dtype=int)

    for agent in range(n_subjects):
        lambda1 = params[agent, 0]
        fishp = generate_fishp(lambda1, n_fish)

        # Predefine block-to-pond assignment, randomized every 3 blocks
        base_sequence = np.array([0, 1, 2] * (n_blocks // 3 + 1))
        block_to_pond = np.random.permutation(base_sequence[:n_blocks])

        for block in range(n_blocks):
            pond_type = block_to_pond[block]  # Get the pond type for this block

            # Generate fish displays for the current pond# Generate fish displays for the current block based on pond distributions
            fish_disp = np.random.choice(
                n_fish, size=n_trials, p=pond_distributions[pond_type]
            )

            pondp = np.ones((n_trials+1, n_fish)) / n_fish
            probabilities[agent, block, 0, :] = pondp[0, :]

            for trial in range(n_trials):
                # observe fish
                ponds[agent, block, trial] = pond_type                
                observations[agent, block, trial] = fish_disp[trial]

                # Bayesian update
                den = np.sum(pondp[trial, :] * fishp[fish_disp[trial], :])
                pondp[trial+1, :] = (fishp[fish_disp[trial], :] * pondp[trial, :]) / den
                pondp[trial+1, :] /= np.sum(pondp[trial+1, :]) # Normalize
                probabilities[agent, block, trial+1, :] = pondp[trial+1, :]

                # choose after update
                choice = np.random.choice(n_fish, p=pondp[trial+1, :])
                choices[agent, block, trial] = choice

    simulated_dict = {
        "params": params,
        "choices": choices,
        "observations": observations,
        "probabilities": probabilities,
        "ponds": ponds
    }
    return simulated_dict

def fit(params, choices, observations, prior=None, output='npl'):
    """
    Fits the model to a single agent's data for one iteration.

    Args:
        params (array): Array containing parameters for the agent (lambda1).
        choices (array): Choices for each agent (n_blocks x n_trials).
        observations (array): Observations for each agent (n_blocks x n_trials).
        prior (optional): Prior distribution to include in calculations.
        output (str): Specifies what to return ('npl' for negative posterior likelihood, 
                      'nll' for negative log-likelihood, or 'all').

    Returns:
        float or dict: Negative log-likelihood/posterior likelihood or detailed results based on output.
    """
    lambda1 = norm2alpha(params[0])

    n_blocks, n_trials = choices.shape
    n_fish = 3  # Assuming a default of 3 fish types; adjust as needed
    fishp = generate_fishp(lambda1, n_fish)
    choice_nll = 0

    # Initialize choice probabilities array for all blocks and trials
    choice_probabilities = np.zeros((n_blocks, n_trials + 1, n_fish))

    for block in range(n_blocks):
        # Initialize pond probabilities for the current block
        pondp = np.ones((n_trials + 1, n_fish)) / n_fish
        choice_probabilities[block, 0, :] = pondp[0, :]

        for trial in range(n_trials):
            fish_disp = observations[block, trial]
            real_choice = choices[block, trial]

            # Bayesian update
            den = np.sum(pondp[trial, :] * fishp[fish_disp, :])
            pondp[trial + 1, :] = (fishp[fish_disp, :] * pondp[trial, :]) / den
            pondp[trial + 1, :] /= np.sum(pondp[trial + 1, :])  # Normalize
            choice_probabilities[block, trial + 1, :] = pondp[trial + 1, :]

            # Accumulate negative log-likelihood for real choices
            choice_nll += -np.log(pondp[trial + 1, real_choice])

    # Calculate the negative posterior likelihood (NPL) if specified
    if output in ('npl', 'nll'):
        fval = calc_fval(choice_nll, params, prior=prior, output=output)
        return fval

    elif output == 'all':
        return {
            "params": np.array([lambda1]),
            "choices": choices,
            "observations": observations,
            "choice_probabilities": choice_probabilities,
            "choice_nll": choice_nll
        }
