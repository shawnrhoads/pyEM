import numpy as np
from tqdm import tqdm
import sys
sys.path.append('../')
from pyEM.math import softmax, norm2beta, norm2alpha

class BehaviorSimulation:
    def __init__(self, params, nblocks=3, ntrials=35, outcomes=None):
        """
        Initialize the behavior simulation.

        Args:
            params: np.array of shape (nsubjects, nparams)
            nblocks: number of blocks to simulate
            ntrials: number of trials per block
            outcomes: optional outcomes to feed into the simulation (default None)
        """
        self.params = params
        self.nblocks = nblocks
        self.ntrials = ntrials
        self.outcomes = outcomes
        
        # Initialize data structures
        nsubjects = params.shape[0]
        self.ev = np.zeros((nsubjects, nblocks, ntrials + 1, 2))
        self.ch_prob = np.zeros((nsubjects, nblocks, ntrials, 2))
        self.choices = np.empty((nsubjects, nblocks, ntrials,), dtype='object')
        self.choices_L = np.zeros((nsubjects, nblocks, ntrials,))
        self.rewards = np.zeros((nsubjects, nblocks, ntrials,))
        self.pe = np.zeros((nsubjects, nblocks, ntrials,))
        self.choice_nll = np.zeros((nsubjects, nblocks, ntrials,))
        
        # Slot probabilities
        self.this_block_probs = [.8, .2]

    def make_choice(self, subj_idx, b, t):
        """Decide which machine to pick."""
        # Calculate choice probability
        self.ch_prob[subj_idx, b, t, :] = softmax(self.ev[subj_idx, b, t, :], self.params[subj_idx, -1]) #last item for beta
        self.choices[subj_idx, b, t] = np.random.choice([0, 1], 
                                                        size=1, 
                                                        p=self.ch_prob[subj_idx, b, t, :])[0]
    
    def get_reward(self, subj_idx, b, t):
        """Obtain the reward based on the choice made."""
        choice = self.choices[subj_idx, b, t]
        if choice == 1:  # Picking the left machine
            self.choices_L[subj_idx, b, t] = 1
            if self.outcomes is None:
                return np.random.choice([1, 0], size=1, p=self.this_block_probs)[0]
            else:
                return self.outcomes[subj_idx][b][t]
        else:
            self.choices_L[subj_idx, b, t] = 0
            if self.outcomes is None:
                return np.random.choice([1, 0], size=1, p=self.this_block_probs[::-1])[0]
            else:
                return self.outcomes[subj_idx][b][t]

    def update_ev(self, subj_idx, b, t, c):
        """Update the expected value based on the PE."""
        # Calculate PE
        self.pe[subj_idx, b, t] = self.rewards[subj_idx, b, t] - self.ev[subj_idx, b, t, c]
        # Update EV for next trial
        self.ev[subj_idx, b, t + 1, :] = self.ev[subj_idx, b, t, :].copy()
        self.ev[subj_idx, b, t + 1, c] = self.ev[subj_idx, b, t, c] + (self.params[subj_idx, 0] * self.pe[subj_idx, b, t]) # 1 for alpha basic model

    def simulate(self):
        """Run the simulation."""
        nsubjects = self.params.shape[0]

        for subj_idx in tqdm(range(nsubjects)):
            # beta, lr = self.params[subj_idx, :]

            for b in range(self.nblocks):
                for t in range(self.ntrials):
                    if t == 0:
                        self.ev[subj_idx, b, t, :] = [.5, .5]

                    # Make choice based on decision policy
                    self.make_choice(subj_idx, b, t)

                    # Get reward
                    self.rewards[subj_idx, b, t] = self.get_reward(subj_idx, b, t)

                    # Get choice index
                    c = int(self.choices[subj_idx, b, t])  # 1 for left, 0 for right

                    # Update EV
                    self.update_ev(subj_idx, b, t, c)

                    # Update negative log-likelihood of choice
                    self.choice_nll[subj_idx, b, t] = self.ch_prob[subj_idx, b, t, c]

        # Return simulation results
        return {
            'params': self.params,
            'ev': self.ev,
            'ch_prob': self.ch_prob,
            'choices': self.choices,
            'choices_L': self.choices_L,
            'rewards': self.rewards,
            'pe': self.pe,
            'choice_nll': self.choice_nll
        }

# Usage Example
params = np.array([[1.0, 0.5], [1.2, 0.4]])  # example parameters for two subjects
simulator = BehaviorSimulation(params)
result = simulator.simulate()

print(result)
