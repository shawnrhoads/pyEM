# Bayesian Inference: Fish Task

Participants observe a sequence of coloured fish and must infer which of three
ponds the fish were drawn from. Each pond contains fish in a fixed 80–10–10
colour ratio. The single parameter ``lambda`` controls how quickly beliefs
update: lower values require more confirming evidence to increase confidence,
making successive observations feel less predictive.

```python
import numpy as np
from pyem.api import EMModel
from pyem.models.bayes import simulate, fit
from pyem.utils.math import norm2alpha

# --- simulate -------------------------------------------------------------
nsubjects, nblocks, ntrials = 30, 6, 15
true_lambda = np.random.uniform(0.2, 0.8, size=(nsubjects, 1))
sim = simulate(true_lambda, n_blocks=nblocks, n_trials=ntrials)
all_data = [[sim["choices"][i], sim["observations"][i]] for i in range(nsubjects)]

# --- fit and recover ------------------------------------------------------
model = EMModel(all_data=all_data, fit_func=fit, param_names=["lambda"])
result = model.fit(verbose=0)
# parameters are in Gaussian space; transform to (0,1)
estimated_lambda = norm2alpha(result.m.T)

corr = np.corrcoef(true_lambda.flatten(), estimated_lambda.flatten())[0, 1]
print("Recovery correlation:", corr)
```
