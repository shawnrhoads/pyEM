# Custom Priors

This example demonstrates fitting a model with a user-defined prior. The
`GaussianPrior` sets independent normal priors for each parameter.

```python
import numpy as np
from pyem import EMModel
from pyem.core.priors import GaussianPrior
from pyem.models.rl import rw1a1b_fit as rw_fit, rw1a1b_simulate as rw_simulate

# simulate data for two subjects
params = np.zeros((2, 2))
sim = rw_simulate(params, nblocks=1, ntrials=4)
all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]

# define a Gaussian prior over parameters
prior = GaussianPrior(mu=[0, 0], sigma=[1, 1])

model = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta", "alpha"])
result = model.fit(prior=prior, verbose=0, njobs=1)
print(result.m)
```

The prior can be customised by implementing the `Prior` protocol and passing the
instance to `EMModel.fit`.

