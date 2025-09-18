<div align="center">

<a target="_blank" rel="noopener noreferrer" href="https://doi.org/10.5281/zenodo.10415396">![DOI:10.5281/zenodo.10415396](https://zenodo.org/badge/DOI/10.5281/zenodo.10415396.svg)</a> <a target="_blank" rel="noopener noreferrer" href="https://github.com/shawnrhoads/pyEM">![GitHub last update](https://img.shields.io/github/last-commit/shawnrhoads/pyEM?color=blue&label=last%20update)</a> [![PyTest](https://github.com/shawnrhoads/pyEM/actions/workflows/pytest.yml/badge.svg)](https://github.com/shawnrhoads/pyEM/actions/workflows/pytest.yml) <a target="_blank" rel="noopener noreferrer" href="https://www.buymeacoffee.com/shawnrhoads">![BuyMeACoffee](https://img.shields.io/static/v1?message=support%20development&label=%20&style=square&logo=Buy%20Me%20A%20Coffee&labelColor=5c5c5c&color=lightgrey)</a>

# pyEM: Expectation Maximization with MAP estimation in Python

</div>

<b>If you use these materials for teaching or research, please use the following citation:</b>
> Rhoads, S. A. (2023). pyEM: Expectation Maximization with MAP estimation in Python. Zenodo. <a href="https://doi.org/10.5281/zenodo.10415396">https://doi.org/10.5281/zenodo.10415396</a>

> Rhoads, S. A., Gan, L., Berluti, K., OConnell, K., Cutler, J., Lockwood, P. L., & Marsh, A. A. (2025). Neurocomputational basis of learning when choices simultaneously affect both oneself and others. In press at *Nature Communications*.

This is a Python implementation of the Hierarchical Expectation Maximization algorithm with MAP estimation for fitting models to behavioral data. [See below](#key-concepts) for more information on the algorithm.

## Quick Start

### Basic Usage

```python
import numpy as np, matplotlib.pyplot as plt
from scipy.stats import truncnorm, beta as beta_dist
from pyem import EMModel
from pyem.utils import plotting
from pyem.utils.math import norm2beta, norm2alpha
from pyem.models.rl import rw1a1b_simulate, rw1a1b_fit

# Settings
nsubjects, nblocks, ntrials = 100, 4, 24
betamin, betamax = .75, 10 # inverse temperature
alphamin, alphamax = .05, .95 # learning rate

# Generate distribution of parameters within range
beta_rv  = truncnorm((betamin-0)/1, (betamax-0)/1, loc=0, scale=2).rvs(nsubjects)
a_lo, a_hi = beta_dist.cdf([alphamin, alphamax], 1.1, 1.1)
alpha_rv = beta_dist.ppf(a_lo + np.random.rand(nsubjects)*(a_hi - a_lo), 1.1, 1.1)

true_params = np.column_stack((beta_rv, alpha_rv))
sim = rw1a1b_simulate(true_params, nblocks=nblocks, ntrials=ntrials)
all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]

# Create and fit model
model = EMModel(
    all_data=all_data,
    fit_func=rw1a1b_fit,
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

# Plot recovery
for param_idx, param_label in enumerate(['beta','alpha']):
    simulated_param = sim['params'][:,param_idx]
    estimated_param = output_dict['params'][:,param_idx]
    ax = plotting.plot_scatter(simulated_param, f'Simulated {param_label}', 
                 estimated_param, f'Estimated {param_label}')
```

### Parameter Recovery Analysis

We can also use the `EMModel.recover()` method to perform parameter recovery directly if you provide the 

```python
import numpy as np, matplotlib.pyplot as plt
from scipy.stats import truncnorm, beta as beta_dist
from pyem import EMModel
from pyem.utils import plotting
from pyem.utils.math import norm2beta, norm2alpha
from pyem.models.rl import rw1a1b_simulate, rw1a1b_fit

# Settings
nsubjects, nblocks, ntrials = 100, 4, 24
betamin, betamax = .75, 10
alphamin, alphamax = .05, .95

# Generate distribution of parameters within range
beta_rv  = truncnorm((betamin-0)/1, (betamax-0)/1, loc=0, scale=2).rvs(nsubjects)
a_lo, a_hi = beta_dist.cdf([alphamin, alphamax], 1.1, 1.1)
alpha_rv = beta_dist.ppf(a_lo + np.random.rand(nsubjects)*(a_hi - a_lo), 1.1, 1.1)

true_params = np.column_stack((beta_rv, alpha_rv))
sim = rw1a1b_simulate(true_params, nblocks=nblocks, ntrials=ntrials)
all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]

# Create model object
model = EMModel(
    all_data=all_data,
    fit_func=rw1a1b_fit,
    param_names=["beta", "alpha"],
    param_xform=[norm2beta, norm2alpha], # Parameter transformation functions
    simulate_func=rw1a1b_simulate
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

```python
import numpy as np
from scipy.stats import truncnorm, beta as beta_dist
from pyem import EMModel
from pyem.core.compare import ModelComparison
from pyem.models.rl import rw1a1b_fit, rw1a1b_simulate, rw2a1b_fit, rw2a1b_simulate
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

rw1a1b_sim = rw1a1b_simulate(true_params, nblocks=nblocks, ntrials=ntrials)
rw1a1b_data = [[c, r] for c, r in zip(rw1a1b_sim["choices"], rw1a1b_sim["rewards"])]

# Create multiple models for comparison
model1 = EMModel(rw1a1b_data, rw1a1b_fit, 
                 param_names=["beta", "alpha"],
                 param_xform=[norm2beta, norm2alpha],
                 simulate_func=rw1a1b_simulate)

model2 = EMModel(rw1a1b_data, rw2a1b_fit,
                 param_names=["beta", "alpha_pos", "alpha_neg"],
                 param_xform=[norm2beta, norm2alpha, norm2alpha],
                 simulate_func=rw2a1b_simulate)

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

We can also compute these metrics individually using the `EMModel` class directly.

```python
# Compute integrated BIC
bicint = model.compute_integrated_bic(nsamples=2000)

# Compute Laplace approximation for Log Model Evidence
lap, lme, good = model.compute_lme()
```

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

#### Example

```python
from pyem import EMModel
from pyem.core.compare import ModelComparison
from pyem.models.rl import rw1a1b_fit, rw1a1b_simulate, rw2a1b_fit, rw2a1b_simulate
from pyem.utils.math import norm2alpha, norm2beta

# Construct two candidate models
model1 = EMModel(all_data, rw1a1b_fit, 
                 param_names=["beta", "alpha"],
                 param_xform=[norm2beta, norm2alpha],
                 simulate_func=rw1a1b_simulate)

model2 = EMModel(all_data, rw2a1b_fit,
                 param_names=["beta", "alpha_pos", "alpha_neg"],
                 param_xform=[norm2beta, norm2alpha, norm2alpha],
                 simulate_func=rw2a1b_simulate)

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

The package includes several pre-implemented models:

### Reinforcement Learning Models (`pyem.models.rl`)

* **`rw1a1b_simulate/fit`**: Rescorla-Wagner model with single learning rate
* **`rw2a1b_simulate/fit`**: Rescorla-Wagner model with separate learning rates for positive/negative prediction errors

### Creating Custom Models

To create a custom model, implement two functions:

```python
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

def my_model_simulate(params, **kwargs):
    """
    Simulation function for your custom model.
    
    Args:
        params: Parameter values in parameter space (n_subjects x n_params)
        **kwargs: Additional simulation parameters
        
    Returns:
        Dictionary with keys (CAN BE ANYTHING): "params", "choices", "rewards", etc.
    """

    # ---- YOUR CODE HERE  ---- 
    # 
    # ------------------------- 

    fval = calc_fval(nll, params, output="nll")

    return {"params": params, "choices": choices, "rewards": rewards}
```

## Key Classes

### EMModel Class

The main interface for model fitting and analysis:

```python
class EMModel:
    def __init__(self, all_data, fit_func, param_names, param_xform=None, simulate_func=None)
    def fit(self, **kwargs) -> FitResult
    def simulate(self, *args, **kwargs)
    def recover(self, true_params, **kwargs) -> dict
    def plot_recovery(self, recovery_dict, **kwargs) -> plt.Figure
    def compute_integrated_bic(self, **kwargs) -> float
    def compute_lme(self) -> tuple
    def get_outfit(self) -> dict
```

### ModelComparison Class

Class for comparing the performance of different models:

```python
class ModelComparison:
    def __init__(self, models, names)
    def compare(self) -> list
    def identify(self) -> pd.DataFrame
    def plot_identifiability(self) -> plt.Figure
```

### Utility Functions

* **Parameter transformations** (`pyem.utils.math`): `norm2alpha()`, `norm2beta()`, `alpha2norm()`, `beta2norm()`
* **Statistics** (`pyem.utils.stats`): `calc_BICint()`, `calc_LME()`, `pseudo_r2_from_nll()`
* **Plotting** (`pyem.utils.plotting`): `plot_scatter()`

## Installation

### Using Anaconda (recommended)

```bash
git clone git+https://github.com/shawnrhoads/pyEM.git
cd pyEM
conda create env --file environment.yml
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

* Python >= 3.10
* numpy >= 1.22
* scipy >= 1.10
* pandas >= 1.5
* matplotlib >= 3.5
* joblib >= 1.3

## Examples

See the `examples/` directory for detailed tutorials:

* `examples/rl.md`: Reinforcement Learning
* `examples/bayes.md`: Bayesian Inference
* `examples/glm.md`: Simple linear modeling

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

<blockquote>Rhoads, S. A., Gan, L., Berluti, K., OConnell, K., Cutler, J., Lockwood, P. L., & Marsh, A. A. (2025). Neurocomputational basis of learning when choices simultaneously affect both oneself and others. In press at Nature Communications. https://doi.org/10.31234/osf.io/rf4x9</blockquote>

See also:
<blockquote>Daw, N. D. (2011). Trial-by-trial data analysis using computational models. Decision making, affect, and learning: Attention and performance XXIII, 23(1). https://doi.org/10.1093/acprof:oso/9780199600434.003.0001 [<a href="https://www.princeton.edu/~ndaw/d10.pdf">pdf</a>]</blockquote>

<blockquote>Huys, Q. J., Cools, R., Gölzer, M., Friedel, E., Heinz, A., Dolan, R. J., & Dayan, P. (2011). Disentangling the roles of approach, activation and valence in instrumental and pavlovian responding. PLoS computational biology, 7(4), e1002028. https://doi.org/10.1371/journal.pcbi.1002028 </blockquote>

**For MATLAB flavors of this algorithm:**
- https://github.com/sjgershm/mfit
- https://github.com/mpc-ucl/emfit
- https://osf.io/s7z6j