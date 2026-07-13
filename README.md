<p align="center">
  <a href="https://shawnrhoadsphd.com/pyEM/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/source/pyem-logo-horizontal-dark-editable.svg">
      <img alt="pyEM" src="assets/source/pyem-logo-horizontal-editable.svg" width="420">
    </picture>
  </a>
</p>

<div align="center">

<a target="_blank" rel="noopener noreferrer" href="https://doi.org/10.5281/zenodo.10415396">![DOI:10.5281/zenodo.10415396](https://zenodo.org/badge/DOI/10.5281/zenodo.10415396.svg)</a> <a target="_blank" rel="noopener noreferrer" href="https://github.com/shawnrhoads/pyEM">![GitHub last update](https://img.shields.io/github/last-commit/shawnrhoads/pyEM?color=blue&label=last%20update)</a> [![PyTest](https://github.com/shawnrhoads/pyEM/actions/workflows/pytest.yml/badge.svg)](https://github.com/shawnrhoads/pyEM/actions/workflows/pytest.yml) <a target="_blank" rel="noopener noreferrer" href="https://creativecommons.org/licenses/by-nc-sa/4.0/">[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/LICENSE-CC%20BY--NC--SA%204.0-teal.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)</a> <a target="_blank" rel="noopener noreferrer" href="https://www.buymeacoffee.com/shawnrhoads">![BuyMeACoffee](https://img.shields.io/static/v1?message=support%20development&label=%20&style=square&logo=Buy%20Me%20A%20Coffee&labelColor=5c5c5c&color=lightgrey)</a>

# pyEM: Expectation Maximization with MAP estimation in Python

</div>

<b>If you use these materials for teaching or research, please use the following citation:</b>
> Rhoads, S. A. (2023). pyEM: Expectation Maximization with MAP estimation in Python. Zenodo. <a href="https://doi.org/10.5281/zenodo.10415396">https://doi.org/10.5281/zenodo.10415396</a>

> Rhoads, S. A., Gan, L., Berluti, K., O'Connell, K., Cutler, J., Lockwood, P. L., & Marsh, A. A. (2025). Neurocomputational basis of learning when choices simultaneously affect both oneself and others. In press at *Nature Communications*. https://doi.org/10.1038/s41467-025-64424-9

pyEM is a Python implementation of Expectation Maximization with MAP for fitting cognitive computational models to behavioral data. [See below](#key-concepts) for more information on the algorithm.

## Quick Start

### Basic Usage

Every model in `pyEM` ships as a pair of `_sim`/`_fit` functions, plus a `ModelSpec` object
(`<name>_model`) that bundles those functions together with the model's identity and description.
Either style works identically — `rw1a1b_model.sim` *is* `rw1a1b_sim`, just accessed through the
bundle:

```python
import numpy as np, matplotlib.pyplot as plt
from scipy.stats import truncnorm, beta as beta_dist
from pyem import EMModel
from pyem.utils import plotting
from pyem.utils.math import norm2beta, norm2alpha
from pyem.models.rl_mf import rw1a1b_model  # bundles rw1a1b_sim, rw1a1b_fit, and metadata
from pyem.core.posterior import parameter_recovery

print(rw1a1b_model.id)      # 'rw1a1b'
print(rw1a1b_model.desc)    # human-readable description
print(rw1a1b_model.spec)    # {'rl': {'softmax': ['beta'], 'rw': ['alpha']}}

# Settings
nsubjects, nblocks, ntrials = 100, 4, 24
betamin, betamax = .75, 10 # inverse temperature
alphamin, alphamax = .05, .95 # learning rate

# Generate distribution of parameters within range
beta_rv  = truncnorm((betamin-0)/1, (betamax-0)/1, loc=0, scale=2).rvs(nsubjects)
a_lo, a_hi = beta_dist.cdf([alphamin, alphamax], 1.1, 1.1)
alpha_rv = beta_dist.ppf(a_lo + np.random.rand(nsubjects)*(a_hi - a_lo), 1.1, 1.1)

true_params = np.column_stack((beta_rv, alpha_rv))
sim = rw1a1b_model.sim(true_params, nblocks=nblocks, ntrials=ntrials)
all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]

# Create and fit model
model = EMModel(
    all_data=all_data,
    fit_func=rw1a1b_model.fit,
    param_names=["beta", "alpha"],
    param_xform=[norm2beta, norm2alpha], # Parameter transformation functions
)

# Fit the model
result = model.fit(verbose=1)
print(f"Convergence: {result.convergence}")

# Access results --> defined from your fit function when `out="all"`
output_dict = model.get_outfit()
estimated_params = output_dict['params']  # Shape: (n_subjects, n_params)
print(f"Estimated parameters shape: {estimated_params.shape}")
print(f"Available outputs: {list(output_dict.keys())}")

# Compare true vs. estimated (matrix)
recovery = parameter_recovery(true_params, estimated_params)
print(f"Correlation per parameter: {recovery.corr}")
print(f"RMSE per parameter: {recovery.rmse}")

# Plot recovery (scatterplot)
for param_idx, param_label in enumerate(['beta','alpha']):
    simulated_param = sim['params'][:,param_idx]
    estimated_param = output_dict['params'][:,param_idx]
    ax = plotting.plot_scatter(simulated_param, f'Simulated {param_label}', 
                 estimated_param, f'Estimated {param_label}')
```

### Using `ModelSpec` with the Parameter Registry

`docs/examples/params.py` (see [Creating Custom Models](#creating-custom-models)) provides a
`build_params()` helper that replaces the hand-rolled `truncnorm`/`beta_dist` calls above with one
call — it returns everything `EMModel` needs (`param_names`, `param_xform`, and natural-space
`true_params`) in one shot, drawn from a shared registry of named, bounded parameters:

```python
import numpy as np
from pyem import EMModel
from pyem.models.rl_mf import rw1a1b_model
from params import build_params  # docs/examples/params.py

# Settings
nsubjects, nblocks, ntrials = 100, 4, 24

# Generate "true" parameters from the shared parameter registry
param_names, param_xform, true_params = build_params(["beta", "alpha"], nsubjects)

# Simulate and fit using the ModelSpec's .sim/.fit directly
sim = rw1a1b_model.sim(true_params, nblocks=nblocks, ntrials=ntrials)
all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]

model = EMModel(
    all_data=all_data,
    fit_func=rw1a1b_model.fit,
    param_names=param_names,
    param_xform=param_xform,
)
result = model.fit(verbose=0)
print(f"Convergence: {result.convergence}")
```

### Parameter Recovery Analysis

We can also use the `EMModel.recover()` method to package the simulate → fit → compare steps above into one call, if you provide any custom simulation and model fitting functions that match [the heuristic below](#creating-custom-models).

```python
import numpy as np, matplotlib.pyplot as plt
from scipy.stats import truncnorm, beta as beta_dist
from pyem import EMModel
from pyem.utils import plotting
from pyem.utils.math import norm2beta, norm2alpha
from pyem.models.rl_mf import rw1a1b_model

# Settings
nsubjects, nblocks, ntrials = 100, 4, 24
betamin, betamax = .75, 10
alphamin, alphamax = .05, .95

# Generate distribution of parameters within range
beta_rv  = truncnorm((betamin-0)/1, (betamax-0)/1, loc=0, scale=2).rvs(nsubjects)
a_lo, a_hi = beta_dist.cdf([alphamin, alphamax], 1.1, 1.1)
alpha_rv = beta_dist.ppf(a_lo + np.random.rand(nsubjects)*(a_hi - a_lo), 1.1, 1.1)

true_params = np.column_stack((beta_rv, alpha_rv))

# Create model object
model = EMModel(
    all_data=None,
    fit_func=rw1a1b_model.fit,
    param_names=["beta", "alpha"],
    param_xform=[norm2beta, norm2alpha], # Parameter transformation functions
    simulate_func=rw1a1b_model.sim,
)

# Perform parameter recovery
recovery_dict = model.recover(
    true_params=true_params,
    pr_inputs=["choices", "rewards"], # inputs needed for fit func
    nblocks=nblocks, ntrials=ntrials # settings for simulate function
)

# Plot recovery results
fig = model.plot_recovery(recovery_dict, figsize=(10, 4))
```

The returned dictionary includes `recovery_dict['correlation']`, an array of
Pearson correlations for each parameter.

### Model Comparison

When we have two different models, we can use the `ModelComparison` class to compare them using various metrics. The package provides several metrics for model comparison:

* **LME** (Log Model Evidence): Laplace approximation to marginal likelihood (formally: log probability of the observed data given a model)
* **Integrated BIC** (Integrated Bayesian Information Criterion): Integrates over the distribution of parameters, which incorporates uncertainty about the parameter values into the model selection process while penalizing model complexity

> [!NOTE]
> The example below calls `display(comparison_df)`, a Jupyter/IPython convenience that is
> injected automatically into the global namespace inside a notebook. If you run this as a plain
> `.py` script (no IPython), replace it with `print(comparison_df)`.

```python
import numpy as np
from scipy.stats import truncnorm, beta as beta_dist
from pyem import EMModel
from pyem.core.compare import ModelComparison
from pyem.models.rl_mf import rw1a1b_model, rw2a1b_model
from pyem.utils.math import norm2alpha, norm2beta

# Settings
nsubjects, nblocks, ntrials = 50, 4, 24
betamin, betamax = .75, 10
alphamin, alphamax = .05, .95

# Generate distribution of parameters within range
beta_rv  = truncnorm((betamin-0)/1, (betamax-0)/1, loc=0, scale=2).rvs(nsubjects)
a_lo, a_hi = beta_dist.cdf([alphamin, alphamax], 1.1, 1.1)
alpha_rv = beta_dist.ppf(a_lo + np.random.rand(nsubjects)*(a_hi - a_lo), 1.1, 1.1)
true_params = np.column_stack((beta_rv, alpha_rv))

sim = rw1a1b_model.sim(true_params, nblocks=nblocks, ntrials=ntrials)
rw1a1b_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]

# Create multiple models for comparison
model1 = EMModel(rw1a1b_data, rw1a1b_model.fit, 
                 param_names=["beta", "alpha"],
                 param_xform=[norm2beta, norm2alpha],
                 simulate_func=rw1a1b_model.sim)

model2 = EMModel(rw1a1b_data, rw2a1b_model.fit,
                 param_names=["beta", "alpha_pos", "alpha_neg"],
                 param_xform=[norm2beta, norm2alpha, norm2alpha],
                 simulate_func=rw2a1b_model.sim)

# Fit both models
res1 = model1.fit(verbose=0)
res2 = model2.fit(verbose=0)

# Compare models
mc = ModelComparison([model1, model2], ["RW1", "RW2"])

bicint_kwargs = {"nsamples":1000, "func_output":"all", "nll_key":"nll"}
r2_kwargs = {"ntrials_total":ntrials*nblocks, "noptions": 2} # two-armed bandit has 2 options
comparison_df = mc.compare(bicint_kwargs=bicint_kwargs, r2_kwargs=r2_kwargs)
display(comparison_df)
```

`compare()` returns a `pandas.DataFrame` indexed by model name, with columns `"LME (largest is
best)"`, `"BICint (smallest is best)"`, and `"pseudoR^2 (largest is best)"` (columns with no
computable values for any model are dropped).

We can also compute these metrics individually using the `EMModel` class directly.

```python
# Compute integrated BIC
bicint = model.compute_integrated_bic(nsamples=2000)

# Compute Laplace approximation for Log Model Evidence
lap, lme, good = model.compute_lme()
```

`compute_integrated_bic()`/`calc_BICint()` need to know how many trials each subject contributed,
for the complexity-penalty term. By default this is auto-detected from your data (correct for
models like the RW/Bayes families, whose data fields are all trial-aligned with identical shape).
If your data doesn't fit that assumption (e.g. a GLM's `[X, Y]`, where `X` has an extra
feature-count dimension), pass it explicitly: `model.compute_integrated_bic(ntrials_total=ntrials*nblocks)`.

### Model Identifiability Analysis

When fitting multiple candidate models to behavioral data, it is crucial to assess **identifiability**, whether simulated data from one model are best recovered by the same model when refitted. `pyEM` provides a convenient interface via the `ModelComparison` class.

Use the `identify()` method to:

1. **Simulate** behavior from each model’s `simulate_func`
2. **Fit** all models to that simulated dataset
3. **Score** each fit using log model evidence (LME), integrated BIC (BICint), and pseudo R²
4. **Count winning models** for each metric across repeated rounds

The result is a `pandas.DataFrame` with per–Simulated/Estimated model entries and summary columns:

* `LME`, `BICint`, `pseudoR2` — mean values across rounds
* `bestlme`, `bestbic`, `bestR2` — number of rounds (0–N) the Estimated model “won” for that metric

You can visualize these results with `plot_identifiability()`, which plots an **asymmetric matrix** (rows = Simulated models, cols = Estimated models) where cell values show the proportion of rounds each Estimated model best fit data from the Simulated model.

> [!NOTE]
> `identify()` draws each round's "true" parameters as raw Gaussian values and maps them into
> natural space via each model's `param_xform` — construct every model with `param_xform` set (as
> shown below), or models with strict natural-space bounds checks (every RW-family model) will
> raise on the untransformed values. `identify()` also requires every model's `simulate_func` to
> return a `dict` of named arrays (true for the RW and Bayes families) — it is **not** currently
> compatible with the GLM family, whose `simulate_func` returns an `(X, Y)` tuple.

#### Example

```python
from pyem import EMModel
from pyem.core.compare import ModelComparison
from pyem.models.rl_mf import rw1a1b_model, rw2a1b_model
from pyem.utils.math import norm2alpha, norm2beta

# Construct two candidate models
model1 = EMModel(all_data, rw1a1b_model.fit, 
                 param_names=["beta", "alpha"],
                 param_xform=[norm2beta, norm2alpha],
                 simulate_func=rw1a1b_model.sim)

model2 = EMModel(all_data, rw2a1b_model.fit,
                 param_names=["beta", "alpha_pos", "alpha_neg"],
                 param_xform=[norm2beta, norm2alpha, norm2alpha],
                 simulate_func=rw2a1b_model.sim)

# Run identifiability analysis
mi_df = mc.identify(
    mi_inputs=['choices','rewards'], # inputs from sim_func required by fit_func
    nrounds=3, # how many rounds to run for MI exercise
    nsubjects=50, # how many computer agents are doing the task? (default: 100)
    sim_kwargs={"nblocks":nblocks, "ntrials": ntrials}, # args for simulate_func
    fit_kwargs={"mstep_maxit": 50},
    verbose=1,
)

print(mi_df.head())
#    Simulated  Estimated  LME  BICint  bestlme  bestbic
# 0  RW1        RW1 ...
# 1  RW1        RW2 ...
# 2  RW2        RW1 ...
# 3  RW2        RW2 ...

# Plot results as proportion of rounds won
mc.plot_identifiability(metric="LME")
mc.plot_identifiability(metric="BICint")
```

### Miscellanous functions

Many computational models have bounded parameters (e.g., learning rates between 0-1). The package uses transformation functions to map between:

* **Normalized space**: Unbounded parameters used during optimization (typically in Gaussian space)
* **Parameter space**: Bounded parameters used in model computations (varies by parameter)

Example:
```python
# Learning rate: 0 ≤ α ≤ 1
alpha_normalized = 0.5  # Unbounded
alpha_natural = norm2alpha(alpha_normalized) # Bounded to [0,1]
```

If you provide function to transform parameters from Gaussian to parameter space (see Daw, 2011), then the following can be used to access them from the `EMModel` class.

```python
# Access parameter transformations
beta_transform = model.get_param_transform("beta")
alpha_transform = model.get_param_transform("alpha")

# Transform parameters
transformed_beta = beta_transform(0.5)  # Convert from normalized to parameter space
transformed_alpha = alpha_transform(0.3)
```

## Available Models

The package includes several pre-implemented models, each described by both its `_sim`/`_fit` functions and a `ModelSpec` (`<name>_model`) carrying its `.id`/`.desc`/`.spec`.

These built-in model families are included for **teaching** and **demonstrations** of this package's flexibility. They can also be used for model fitting in your own research, but please note that they use specific task structures that might not match your own study design or research question. I am happy to chat if you have any questions about this. You can easily create your own custom model by following the [custom model template](#creating-custom-models). pyEM also provides utilities for model comparison and parameter recovery, which can be used to evaluate the identifiability of your own models.

### Linear Models (`pyem.models.glm`)

* **`glm_sim/fit`** (id: `glm`): standard Gaussian linear regression. Free parameters: regression weights (intercept + covariates).
* **`glm_decay_sim/fit`** (id: `glm_decay`): Gaussian linear regression with exponentially discounted predictors. Free parameters: regression weights, `gamma` (discount factor, in `[0,1]`).
* **`logit_sim/fit`** (id: `logit`): standard logistic regression. Free parameters: regression weights (intercept + covariates).
* **`logit_decay_sim/fit`** (id: `logit_decay`): logistic regression with exponentially discounted predictors. Free parameters: regression weights, `gamma`.
* **`glm_ar_sim/fit`** (id: `glm_ar`): Gaussian linear regression with an AR(1) autoregressive term on the residuals. Free parameters: regression weights, `phi` (AR(1) coefficient, in `(-1,1)`).

### Model-Free Reinforcement Learning Models (`pyem.models.rl_mf`)

* **`rw1a1b_sim/fit`** (id: `rw1a1b`): Rescorla-Wagner model with a single learning rate. Free parameters: `beta`, `alpha`.
* **`rw2a1b_sim/fit`** (id: `rw2a1b`): Rescorla-Wagner model with separate learning rates for positive vs. negative prediction errors (valence bias). Free parameters: `beta`, `alpha_pos`, `alpha_neg`.
* **`rw3a1b_sim/fit`** (id: `rw3a1b`): two-option task with three binary outcome channels (self/other/no one); combines self/other/no-one prediction errors into a single expected-value update ([Lockwood et al., 2016](https://doi.org/10.1073/pnas.1603198113)). Free parameters: `beta`, `alpha_self`, `alpha_other`, `alpha_noone`.
* **`rw4a1b_sim/fit`** (id: `rw4a1b`): four-option task where each trial shows a pair of options; one shared inverse temperature and four learning rates split by outcome recipient (self/other) and valence (positive/negative) ([Rhoads et al., 2025](https://doi.org/10.1038/s41467-025-64424-9)). Free parameters: `beta`, `alpha_self_pos`, `alpha_self_neg`, `alpha_other_pos`, `alpha_other_neg`.

### Model-Based Reinforcement Learning Models (`pyem.models.rl_mb`)

Three learners for the Daw et al. (2011) two-step task ([Daw et al., 2011](https://doi.org/10.1016/j.neuron.2011.02.027)). All fit `beta1`/`beta2` in `(0, inf)` via `exp(x)`, the learning rates / trace / weight in `[0,1]`, and the first-stage stickiness `r` in `(-inf, inf)`.

* **`sarsa_lambda_sim/fit`** (id: `sarsa_lambda`): model-free SARSA(&lambda;) learner (`omega = 0`). Free parameters: `beta1`, `beta2`, `alpha1`, `alpha2`, `lambda`, `r`.
* **`model_based_sim/fit`** (id: `model_based`): model-based Bellman learner (`omega = 1`; `alpha1` and `lambda` drop out). Free parameters: `beta1`, `beta2`, `alpha2`, `r`.
* **`hybrid_mbmf_sim/fit`** (id: `hybrid_mbmf`): hybrid that mixes model-based and model-free first-stage values with weight `omega` (Daw's `w`). Free parameters: `beta1`, `beta2`, `alpha1`, `alpha2`, `lambda`, `omega`, `r`.

### Bayesian Inference (`pyem.models.bayes`)

* **`bayes_sim/fit`** (id: `bayes`): Bayesian belief-updating over which of three sources (e.g. "ponds") an observation came from, given no feedback. Free parameter: `lambda1` (belief-update rate, in `[0,1]`).

### Discounting Models (`pyem.models.discounting`)

Five discounting domains, all sharing the same shape: a block-level discounting variable (social
distance, delay, odds against, or effort level) discounts one option's utility, and choice follows
a logistic rule on the resulting value difference (`sigmoid(delta_V)`).

* **`sd_hyp_wk_sim/fit`** (id: `sd_hyp_wk`): hyperbolic social discounting with a free other-regarding weight, `U_other(N) = w_other*r_other / (1 + k*N)`. Free parameters: `w_other`, `k`.
* **`sd_hyp_k_sim/fit`** (id: `sd_hyp_k`): hyperbolic social discounting with the weight fixed at 1, `U_other(N) = r_other / (1 + k*N)`. Free parameter: `k`.
* **`sd_par_k_sim/fit`** (id: `sd_par_k`): parabolic social discounting, `U_other(N) = r_other - k*N**2`. Free parameter: `k`.
* **`sd_lin_k_sim/fit`** (id: `sd_lin_k`): linear social discounting, `U_other(N) = r_other - k*N`. Free parameter: `k`.
* **`td_hyp_k_sim/fit`** (id: `td_hyp_k`): hyperbolic temporal (delay) discounting ([Mazur, 1987](https://doi.org/10.4324/9781315798260)) in a smaller-sooner vs. larger-later choice. Free parameter: `k`.
* **`prd_hyp_k_sim/fit`** (id: `prd_hyp_k`): hyperbolic probability discounting ([Rachlin, Raineri, & Cross, 1991](https://doi.org/10.1901/jeab.1991.55-233)). Free parameter: `k`.
* **`ed_par_k_sim/fit`** (id: `ed_par_k`): parabolic effort discounting (accelerating effort cost). Free parameter: `k`.
* **`ped_par_k_sim/fit`** (id: `ped_par_k`): parabolic prosocial effort discounting with a single discount rate shared across self/other. Free parameter: `k`.
* **`ped_par_2k_sim/fit`** (id: `ped_par_2k`): parabolic prosocial effort discounting with separate self/other discount rates. Free parameters: `k_self`, `k_other`.

See `docs/examples/discounting.ipynb` for worked examples of all nine variants.

### Prospect Theory (`pyem.models.pt`)

* **`pt_sim/fit`** (id: `pt`): Prospect Theory ([Tversky & Kahneman, 1992](https://doi.org/10.1007/BF00122574)) model of choices between a certain amount and a two-outcome gamble; a power value function with separate gain/loss curvature and a loss-aversion multiplier, combined with a one-parameter probability weighting function and a logistic choice rule. Free parameters: `alpha` (gain curvature), `beta` (loss curvature), `lambda` (loss aversion), `gamma` (probability weighting), `mu` (choice temperature).

### Signal Detection Theory (`pyem.models.sdt`)

* **`sdt_sim/fit`** (id: `sdt`): equal-variance Gaussian signal detection theory model of an old/new recognition memory task. Free parameters: `dprime` (sensitivity, `d' >= 0`), `criterion` (response bias, `c`).

### Drift-Diffusion Model (`pyem.models.ddm`)

* **`ddm4_sim/fit`** (id: `ddm4`): four-parameter DDM of a HIGH-vs-LOW value choice (two certain amounts per trial, choose the higher); a two-boundary Wiener diffusion (upper = high/correct, lower = low/error) with the Navarro & Fuss (2009) WFPT likelihood; drift v = v_coef*(value_high - value_low). Free parameters: `v_coef`, `a`, `t0`, `z`. Across-trial variability `sv`, `st`, `sz` (drift SD, non-decision-time width, start-point width) are set to 0, marginalized analytically for `sv` (Ratcliff & Tuerlinckx, 2002) and by Gauss-Legendre quadrature for `st`, `sz.
* **`ddm4_lotto_sim/fit`** (id: `ddm4_lotto`): four-parameter DDM of a safe-vs-risky GAMBLE (risky gamble EV=p*payoff vs safe certain amount); a two-boundary Wiener diffusion (upper = risky, lower = safe) with the Navarro & Fuss (2009) WFPT likelihood; drift v = v_coef*(EV_risky - safe). Free parameters: `v_coef`, `a`, `t0`, `z`. Across-trial variability `sv`, `st`, `sz` (drift SD, non-decision-time width, start-point width) are set to 0, marginalized analytically for `sv` (Ratcliff & Tuerlinckx, 2002) and by Gauss-Legendre quadrature for `st`, `sz.

### Creating Custom Models

Every model above follows the same template: a pair of `_sim`/`_fit` functions plus a `ModelSpec`
that bundles them with a hand-authored id/description/spec. To create a custom model, follow the
same shape:

> [!NOTE]
> Both `EMModel.fit()`'s EM loop and `model.recover()` always call your `_fit` function with
> `prior=...` passed as a **keyword** argument (never positionally). This works whether your
> function lists its data arguments explicitly (`params, choices, rewards, *, prior=None,
> output="npl"`, as below) or captures them variadically (`params, *data, prior=None,
> output="npl"` — `*data` already forces everything after it to be keyword-only, so no extra `*`
> is needed in that case). Either keyword-only style is safe to use with `EMModel.fit`.

```python
from pyem.core.modelspec import ModelSpec
from pyem.utils.math import norm2alpha, norm2beta, calc_fval

def my_model_fit(params, choices, rewards, *, prior=None, output="npl"):
    """Fit function for your custom model.

    Parameters
    ----------
    params : sequence
        Parameter values in normalized space.
    choices, rewards : sequence
        Subject-specific data passed to the model.
    prior : object, optional
        Prior distribution with ``logpdf`` method.
    output : {"npl", "nll", "all"}
        Determines the function output.

    Returns
    -------
    float or dict
        Objective value or full output when ``output='all'``.
    """
    # ---- EDIT AS NEEDED  ---- 
    alpha = norm2alpha(params[0])
    beta = norm2beta(params[1])

    if not (0 <= alpha <= 1):
        return 1e7
    if not (0.001 <= beta <= 20):
        return 1e7
    # ------------------------- 

    # ---- YOUR CODE HERE  ---- 
    # Model-specific negative log-likelihood
    nll = ...
    # ------------------------- 

    if output == "all":
        return {"params": [alpha, beta], "choices": choices, "rewards": rewards, "nll": nll} # should have "params" and "nll"

    return calc_fval(nll, params, prior=prior, output=output)

def my_model_sim(params, **kwargs):
    """
    Simulation function for your custom model.
    
    Args:
        params: Parameter values in parameter space (n_subjects x n_params)
        **kwargs: Additional simulation parameters
        
    Returns:
        Dictionary with keys (CAN BE ANYTHING): "params", "choices", "rewards", etc.
    """

    # ---- YOUR CODE HERE  ---- 
    # Simulate trials/choices/rewards for your task and populate output arrays
    # ------------------------- 

    # (Optional) If you want to compute a likelihood value for debugging, define
    # it explicitly here; otherwise remove this line.
    # fval = calc_fval(nll, params, output="nll")

    return {"params": params, "choices": choices, "rewards": rewards}

# Wrapper metadata, following the same shape as every model in pyem.models.*
my_model_desc = "One-sentence description of what this model does and its free parameters."
my_model_id = "my_model"
my_model_spec = {"rl": {"softmax": ["beta"], "rw": ["alpha"]}}  # free-form, hand-authored — no fixed vocabulary to satisfy
my_model = ModelSpec(
    id=my_model_id, spec=my_model_spec, desc=my_model_desc,
    params=None, sim=my_model_sim, fit=my_model_fit,
)
```

`ModelSpec` is deliberately a plain, unopinionated container (`id`, `spec`, `desc`, `params`,
`sim`, `fit`) — it imposes no naming scheme on your parameters or spec taxonomy, and nothing in
`EMModel`/`ModelComparison` requires you to build one at all; it exists purely so a model can
describe itself.

If you'd like a reusable pattern for generating named, bounded "true" parameters for simulation
(rather than hand-rolling `truncnorm`/`beta_dist` calls every time, as the examples above do), see
`docs/examples/params.py`. It defines a `ParamDef`/`PARAM_REGISTRY` pattern
(`build_params(["beta", "alpha"], nsubjects)` → `(param_names, param_xform, true_params)`) that the
example notebooks use. This lives in `docs/examples/`, not the installed package, by design — `pyem`
itself stays agnostic about what parameters any given model uses, so this is a copyable starting
point for your own model collection rather than a package dependency.

## Key Classes

### EMModel Class

The main interface for model fitting and analysis:

```python
class EMModel:
    def __init__(self, all_data, fit_func, param_names, param_xform=None, simulate_func=None)
    def fit(self, **kwargs) -> FitResult
    def simulate(self, *args, **kwargs)
    def recover(self, true_params, pr_inputs, **kwargs) -> dict
    def plot_recovery(self, recovery_dict, **kwargs) -> plt.Figure
    def subject_params(self) -> np.ndarray
    def compute_integrated_bic(self, **kwargs) -> float
    def compute_lme(self) -> tuple
    def get_outfit(self) -> dict
```

`subject_params()` is the recommended way to get each subject's fitted parameters in natural
space — it applies `param_xform` for you, the same way `get_outfit()['params']` does when your fit
function's `output="all"` branch returns a `'params'` key.

### ModelSpec Class

A plain dataclass bundling a model's identity with its entry points (see "Creating Custom Models"
above):

```python
class ModelSpec:
    id: str
    spec: dict
    desc: str
    params: Callable | None
    sim: Callable
    fit: Callable
```

### ModelComparison Class

Class for comparing the performance of different models:

```python
class ModelComparison:
    def __init__(self, models, model_names=None)
    def compare(self, **kwargs) -> pd.DataFrame
    def identify(self, mi_inputs, nrounds=10, nsubjects=100, **kwargs) -> pd.DataFrame
    def plot_identifiability(self, **kwargs) -> plt.Figure
```

### Utility Functions

* **Parameter transformations** (`pyem.utils.math`): `norm2alpha()`, `norm2beta()`, `alpha2norm()`, `beta2norm()`
* **Statistics** (`pyem.utils.stats`): `calc_BICint()`, `calc_LME()`, `pseudo_r2_from_nll()`
* **Plotting** (`pyem.utils.plotting`): `plot_scatter()`
* **Parameter registry** (`docs/examples/params.py`, not part of the installed package): `ParamDef`, `PARAM_REGISTRY`, `build_params()`, `validate_params()`

## EM & Optimizer Configuration

`EMModel.fit()` exposes two layers of configuration, matching the two nested loops that make up
hierarchical EM:

* **Outer EM loop** (`EMConfig`): alternates an E-step (fit each subject given the current
  population-level prior) and an M-step (update the population-level prior from all subjects'
  fits) until convergence.
* **Inner per-subject optimizer** (`OptimConfig`): the `scipy.optimize.minimize` call used
  *inside* each subject's E-step to find that subject's MAP parameter estimate.

You never construct `EMConfig`/`OptimConfig` directly — `fit()` builds them for you from its
keyword arguments:

```python
result = model.fit(
    mstep_maxit=200,             # EMConfig: max outer EM iterations
    convergence_method="sum",    # EMConfig: "sum" | "mean" | "median" aggregation of per-subject deltas
    convergence_custom=None,     # EMConfig: None | "relative_npl" | "running_average"
    convergence_crit=1e-3,       # EMConfig: convergence threshold
    convergence_precision=6,     # EMConfig: decimal precision for convergence comparisons
    njobs=-2,                    # EMConfig: parallel jobs (joblib) across subjects within an E-step
    seed=None,                   # EMConfig: RNG seed for the EM loop / optimizer restarts
    mstep="gaussian",            # EMConfig: population-distribution family fit by the M-step
    optim_method="BFGS",         # OptimConfig: scipy.optimize.minimize method for each subject
    optim_options=None,          # OptimConfig: extra options merged into {"gtol": 1e-4, "eps": 1e-4}
    max_restarts=2,              # OptimConfig: extra random-restart attempts if not successful
)
```

> [!NOTE]
> `OptimConfig` also has an `x_scale` field (scale of the random initial guess drawn for each
> optimizer attempt; default `0.1`), but `fit()` does not currently expose a matching keyword for
> it — its default always applies when fitting through `EMModel`.

### M-step distribution families (`mstep=...`)

By default the M-step fits a Gaussian population distribution per parameter (`mstep="gaussian"`),
matching the classic Huys/Daw hierarchical EM formulation. Heavier-tailed alternatives can improve
robustness to outlier subjects:

```python
result = model.fit(mstep="laplace")
```

* `"gaussian"` (default): Normal population distribution.
* `"laplace"`: Laplace (double-exponential) distribution — heavier tails than Gaussian.
* `"student_t"`: Student's t distribution (default 8 degrees of freedom) — heavier tails still.
* `"cauchy"`: Cauchy distribution (Student's t with 1 degree of freedom) — very heavy tails.

## Installation

### Using Anaconda (recommended)

```bash
git clone https://github.com/shawnrhoads/pyEM.git
cd pyEM
conda env create --file environment.yml
```

### Using GitHub
```bash
pip install git+https://github.com/shawnrhoads/pyEM.git
```

### For Development

```bash
git clone https://github.com/shawnrhoads/pyEM.git
cd pyEM
pip install -e .
```

## Requirements

Core (installed automatically with `pip install pyem` or `pip install -e .`):

* Python >= 3.10
* numpy >= 1.22
* scipy >= 1.10
* pandas >= 1.5
* matplotlib
* joblib >= 1.3
* typing-extensions >= 4.6

Optional extras (installed via `pip install`'s extras syntax):

* `pyem[viz]`: `seaborn` — used by some plotting helpers (imported lazily; not required for core `EMModel`/`ModelComparison` usage).
* `pyem[extras]`: `statsmodels`, `scikit-learn`, `tqdm` — used by some example notebooks.
* `pyem[dev]`: `pytest` — for running the test suite.

To install everything needed to run the example notebooks:

```bash
pip install 'pyem[viz,extras]'
```

## Examples

See the `docs/examples/` directory for detailed tutorials:

* `docs/examples/rl_mf.ipynb`: Model-Free Reinforcement Learning — free params `beta`, `alpha` (and variants: `alpha_pos`/`alpha_neg`, `alpha_self`/`alpha_other`/`alpha_noone`, `alpha_self_pos`/`alpha_self_neg`/`alpha_other_pos`/`alpha_other_neg`)
* `docs/examples/rl_mb.ipynb`: Model-Based Reinforcement Learning — Daw two-step task, three models (`sarsa_lambda`, `model_based`, `hybrid_mbmf`), free params `beta1`, `beta2`, `alpha1`, `alpha2`, `lambda`, `omega`, `r`
* `docs/examples/bayes.ipynb`: Bayesian Inference — free param `lambda1`
* `docs/examples/glm.ipynb`: Simple linear modeling — free params: regression weights (plus `gamma` for `*_decay` variants, `phi` for `glm_ar`)
* `docs/examples/discounting.ipynb`: Social/temporal/probability/effort discounting — free params `w_other`, `k` (or `k_self`/`k_other` for the prosocial-effort model), see [Discounting Models](#discounting-models-pyemmodelsdiscounting)
* `docs/examples/pt.ipynb`: Prospect Theory — free params `alpha`, `beta`, `lambda`, `gamma`, `mu`
* `docs/examples/sdt.ipynb`: Signal Detection Theory — free params `dprime`, `criterion`
* `docs/examples/ddm.ipynb`: Drift-Diffusion Models — high-vs-low value and safe-vs-risky gamble tasks, four models (ddm4/ddm4_lotto), parameter recovery + across-trial-variability visualizations

## Documentation

The full documentation site — API reference, guides, and the rendered example
notebooks — is built with [MkDocs](https://www.mkdocs.org/) and the
[Material](https://squidfunk.github.io/mkdocs-material/) theme, and is published at
**https://shawnrhoadsphd.com/pyEM/**.

### Building and viewing the docs locally

From the repository root:

```bash
# 1) Install the package (so the API reference can import pyem) + the doc toolchain
pip install -e .
pip install -r requirements-docs.txt

# 2) Live preview with auto-reload — open the URL it prints (default http://127.0.0.1:8000/)
mkdocs serve

# 3) …or build the static site into ./site/ (this is what CI runs;
#    --strict turns any warning into an error)
mkdocs build --strict
```

The doc dependencies are version-pinned in `requirements-docs.txt` for reproducible
builds, and the notebooks under `docs/examples/` are rendered directly by
`mkdocs-jupyter`.

## Testing

Run the test suite:

```bash
pytest tests/
```

## For Contributors
This is meant to be a basic implementation of hierarchical EM with MAP estimation, but I invite other researchers and educators to help improve and expand the code here!

Here are some ways you can help!
- If you spot an error (e.g., typo, bug, inaccurate descriptions, etc.), please open a new issue on GitHub by clicking on the GitHub Icon in the top right corner on any page and selecting "open issue". Alternatively, you can <a target="_blank" rel="noopener noreferrer" href="https://github.com/shawnrhoads/pyEM/issues/new?labels=bug&template=issue-template.yml">open a new issue</a> directly through GitHub.
- If there is inadvertently omitted credit for any content that was generated by others, please also <a target="_blank" rel="noopener noreferrer" href="https://github.com/shawnrhoads/pyEM/issues/new?labels=enhancement&template=enhancement-template.yml">open a new issue</a> directly through GitHub.
- If you have an idea for a new example tutorial or a new module to include, please either <a target="_blank" rel="noopener noreferrer" href="https://github.com/shawnrhoads/pyEM/issues/new?labels=enhancement&template=enhancement-template.yml">open a new issue</a> and/or submit a pull request directly to the repository on GitHub.

<hr>

## Key Concepts

The algorithm fits models using a hierarchical approach where:

1. **E-step**: Estimates subject-specific parameters given population-level priors
2. **M-step**: Updates population-level priors given subject-specific parameters
3. **MAP estimation**: Incorporates prior beliefs into likelihood to regularize parameter estimates

*Negative Log-Likelihood* 

The negative log-likelihood is a measure of how well the model fits the observed data. It is obtained by taking the negative natural logarithm of the likelihood function. The goal of MLE is to find the parameter values that minimize the negative log-likelihood, effectively maximizing the likelihood of the observed data given the model.

*Prior Probability*

The prior probability represents our knowledge or belief about the parameters before observing the data. It is typically based on some prior information or assumptions. In this case, we are using a normal distribution to represent our prior beliefs about the parameters, with mean $\mu$ and standard deviation $\sqrt{\sigma}$.

*MAP Estimation*

In MAP estimation, we are incorporating the prior probability into the estimation process. Instead of only maximizing the likelihood (as in MLE), we are maximizing the posterior probability, which combines the likelihood and the prior. Mathematically, MAP estimation can be expressed as: 

$argmax_{\theta} (likelihood(data | \theta) * prior(\theta))$

where $\theta$ represents the model parameters

We are effectively combining the likelihood and the prior in a way that biases the parameter estimation towards the prior beliefs. Since we are maximizing this combined term, we are seeking parameter values that not only fit the data well (as indicated by the likelihood) but also align with the prior probability distribution.

**Code originally adapted for Python from:**
<blockquote>Wittmann, M. K., Fouragnan, E., Folloni, D., Klein-Flügge, M. C., Chau, B. K., Khamassi, M., & Rushworth, M. F. (2020). Global reward state affects learning and activity in raphe nucleus and anterior insula in monkeys. Nature Communications, 11(1), 3771. https://doi.org/10.1038/s41467-020-17343-w</blockquote>

<blockquote>Cutler, J., Wittmann, M. K., Abdurahman, A., Hargitai, L. D., Drew, D., Husain, M., & Lockwood, P. L. (2021). Ageing is associated with disrupted reinforcement learning whilst learning to help others is preserved. Nature Communications, 12(1), 4440. https://doi.org/10.1038/s41467-021-24576-w</blockquote>

<blockquote>Rhoads, S. A., Gan, L., Berluti, K., O'Connell, K., Cutler, J., Lockwood, P. L., & Marsh, A. A. (2025). Neurocomputational basis of learning when choices simultaneously affect both oneself and others. Nature Communications. 16, 9350. https://doi.org/10.1038/s41467-025-64424-9</blockquote>

See also:
<blockquote>Daw, N. D. (2009). Trial-by-trial data analysis using computational models. Decision making, affect, and learning: Attention and performance XXIII, 23(1). https://doi.org/10.1093/acprof:oso/9780199600434.003.0001 [<a href="https://www.princeton.edu/~ndaw/d10.pdf">pdf</a>]</blockquote>

<blockquote>Huys, Q. J., Cools, R., Gölzer, M., Friedel, E., Heinz, A., Dolan, R. J., & Dayan, P. (2011). Disentangling the roles of approach, activation and valence in instrumental and pavlovian responding. PLoS computational biology, 7(4), e1002028. https://doi.org/10.1371/journal.pcbi.1002028 </blockquote>

**For MATLAB flavors of this algorithm:**
- https://github.com/sjgershm/mfit
- https://github.com/mpc-ucl/emfit
- https://osf.io/s7z6j
