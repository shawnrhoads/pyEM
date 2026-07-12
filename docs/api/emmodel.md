# EMModel & FitResult

`EMModel` is the high-level, scikit-learn-style wrapper around the hierarchical
EM fit. You give it a list of per-subject data, an objective (`fit_func`), the
parameter names, and (optionally) the Gaussian→natural transforms and a
simulator. `EMModel.fit()` returns a `FitResult` dataclass. Parameters are fit in
**Gaussian (unbounded) space**; `subject_params()` returns natural-space estimates
when `param_xform` is provided.

```python
import numpy as np
from pyem import EMModel
from pyem.models.rl_mf import rw1a1b_sim, rw1a1b_fit
from pyem.utils.math import norm2beta, norm2alpha

rng = np.random.default_rng(0)
true = np.column_stack([norm2beta(rng.normal(size=20)), norm2alpha(rng.normal(size=20))])
sim = rw1a1b_sim(true, nblocks=3, ntrials=24, seed=0)
all_data = [[sim["choices"][s], sim["rewards"][s]] for s in range(20)]

model = EMModel(all_data, rw1a1b_fit, ["beta", "alpha"],
                param_xform=[norm2beta, norm2alpha], simulate_func=rw1a1b_sim)
result = model.fit(verbose=0)
print(result.m.shape, result.convergence)     # (2, 20) True/False
print(model.subject_params().shape)           # (20, 2) natural-space
```

::: pyem.api.EMModel

::: pyem.api.FitResult
