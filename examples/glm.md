# General Linear Model

This example shows how to simulate and recover parameters for a simple linear
regression model.  Each subject has their own regression coefficients and we fit
all subjects simultaneously using ``EMModel``.

```python
import numpy as np
from pyem import EMModel
from pyem.models.glm import simulate, fit

# --- simulate -------------------------------------------------------------
nsubjects, nparams, ntrials = 40, 3, 100
true_params = np.random.randn(nsubjects, nparams)
X, Y = simulate(true_params, ntrials=ntrials)
all_data = [[X[i], Y[i]] for i in range(nsubjects)]

# --- fit and recover ------------------------------------------------------
model = EMModel(all_data=all_data, fit_func=fit,
                param_names=[f"b{i}" for i in range(nparams)])
result = model.fit(verbose=0)
estimated = result.m.T

# simple recovery metrics per parameter
corr = np.array([np.corrcoef(true_params[:, j], estimated[:, j])[0, 1]
                 for j in range(nparams)])
print("Recovery correlations:", corr)
```
