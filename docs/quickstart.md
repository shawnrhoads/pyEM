# Quick Start

This walkthrough simulates data from a simple Rescorla-Wagner reinforcement-learning model, fits it back with [`EMModel`](api/emmodel.md), and inspects the resulting group- and subject-level estimates. It uses the same `rw1a1b` model family shown in the [Model-Free Reinforcement Learning example](examples/rl_mf.ipynb).

```python
import numpy as np
from pyem import EMModel
from pyem.models.rl_mf import rw1a1b_sim, rw1a1b_fit
from pyem.utils.math import norm2beta, norm2alpha

# 1) Simulate 30 subjects of a 1-learning-rate Rescorla–Wagner task
rng = np.random.default_rng(0)
true_params = np.column_stack([
    norm2beta(rng.normal(size=30)),    # inverse temperature (beta)
    norm2alpha(rng.normal(size=30)),   # learning rate (alpha)
])
sim = rw1a1b_sim(true_params, nblocks=3, ntrials=24, seed=0)

# 2) Package each subject's data as [choices, rewards]
all_data = [[sim["choices"][s], sim["rewards"][s]] for s in range(30)]

# 3) Build the hierarchical model and fit it
model = EMModel(
    all_data=all_data,
    fit_func=rw1a1b_fit,
    param_names=["beta", "alpha"],
    param_xform=[norm2beta, norm2alpha],
    simulate_func=rw1a1b_sim,
)
result = model.fit(verbose=0)

# 4) Inspect results
print("MAP params per subject:", model.subject_params().shape)   # (30, 2)
print("Group posterior mean:", result.posterior_mu)
print("Converged:", result.convergence)
```

Walking through the steps:

1. **Simulate.** `true_params` are drawn in unbounded **Gaussian space** (the space the EM optimizer works in) and mapped to natural-space values with `norm2beta` (inverse temperature, bounded to `(0, 20)`) and `norm2alpha` (learning rate, bounded to `(0, 1)`). `rw1a1b_sim` takes natural-space parameters and simulates choices and rewards for each subject across blocks and trials.
2. **Package the data.** `EMModel` expects `all_data` to be a list with one entry per subject; each subject's entry is the exact list of positional arguments (after `params`) that the model's fit function expects — here `[choices, rewards]`, matching `rw1a1b_fit(params, choices, rewards, ...)`.
3. **Fit.** `EMModel(...)` bundles the data, the fit function, the parameter names, and the `param_xform` list (the same `norm2beta`/`norm2alpha` transforms used to simulate) so that the optimizer can search in Gaussian space while reporting natural-space values. `.fit()` runs the EM loop and returns a `FitResult`.
4. **Inspect.** `model.subject_params()` returns MAP estimates in natural space, one row per subject. `result.posterior_mu` is the fitted group-level mean (in Gaussian space), and `result.convergence` reports whether the EM loop converged.
