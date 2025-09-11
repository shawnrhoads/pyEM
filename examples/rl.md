# Two-Armed Bandit Reinforcement Learning

This example demonstrates parameter recovery for a simple two-armed bandit
reinforcement learning task.  On each trial the participant chooses between two
options with different reward probabilities.  Choices are generated with a
Rescorla–Wagner model using an inverse-temperature parameter (``beta``) and a
learning rate (``alpha``).

```python
import numpy as np
from pyem.api import EMModel
from pyem.models.rl import rw1a1b_simulate, rw1a1b_fit

# --- simulate subjects ----------------------------------------------------
nsubjects, nblocks, ntrials = 50, 6, 24
true_params = np.column_stack([
    np.random.randn(nsubjects),  # beta in Gaussian space
    np.random.randn(nsubjects),  # alpha in Gaussian space
])

# use EMModel.recover to run simulation, fitting and recovery metrics
model = EMModel(all_data=None, fit_func=rw1a1b_fit,
                param_names=["beta", "alpha"],
                simulate_func=rw1a1b_simulate)
recovery = model.recover(true_params, nblocks=nblocks, ntrials=ntrials)

# scatter plot of recovered parameters
fig = model.plot_recovery(recovery)
```

The recovery dictionary also contains numerical summaries such as
`recovery['correlation']`.
