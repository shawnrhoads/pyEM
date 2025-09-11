<div align="center">

<a target="_blank" rel="noopener noreferrer" href="https://doi.org/10.5281/zenodo.10415396">![DOI:10.5281/zenodo.10415396](https://zenodo.org/badge/DOI/10.5281/zenodo.10415396.svg)</a> <a target="_blank" rel="noopener noreferrer" href="https://github.com/shawnrhoads/pyEM">![GitHub last update](https://img.shields.io/github/last-commit/shawnrhoads/pyEM?color=blue&label=last%20update)</a> <a target="_blank" rel="noopener noreferrer" href="https://www.buymeacoffee.com/shawnrhoads">![BuyMeACoffee](https://img.shields.io/static/v1?message=contribute%20caffeine&label=%20&style=square&logo=Buy%20Me%20A%20Coffee&labelColor=5c5c5c&color=lightgrey)</a>

# pyEM: Expectation Maximization with MAP estimation in Python

</div>

<b>If you use these materials for teaching or research, please use the following citation:</b>
> Rhoads, S. A. (2023). pyEM: Expectation Maximization with MAP estimation in Python. Zenodo. <a href="https://doi.org/10.5281/zenodo.10415396">https://doi.org/10.5281/zenodo.10415396</a>

This is a Python implementation of the Hierarchical Expectation Maximization algorithm with MAP estimation for fitting models to behavioral data. [See below](#key-concepts) for more information on the algorithm.

## Features

* **High-level API**: The `EMModel` class provides a sklearn-like interface for model fitting and analysis
* **Parameter recovery**: Built-in methods for parameter recovery analysis and visualization
* **Model comparison**: Tools for comparing different models using LME, BIC, and integrated BIC
* **Parameter transformations**: Support for parameter transformation functions to handle bounded parameters
* **Modular design**: Easy to extend with custom models and analysis methods
* **Comprehensive testing**: Extensive test coverage for reliability

## Quick Start

### Basic Usage

```python
import numpy as np
import matplotlib.pyplot as plt
from pyem.api import EMModel
from pyem.models.rl import rw1a1b_simulate, rw1a1b_fit
from pyem.utils.math import norm2beta, norm2alpha
from pyem.utils import plotting

# Generate sample data
nsubjects, nblocks, ntrials = 100, 6, 24
true_params = np.column_stack([np.random.normal(-1.5,1,nsubjects), 
                               np.random.normal(0,1,nsubjects)]) # Untransformed parameters in Gaussian space
sim = rw1a1b_simulate(true_params, nblocks=3, ntrials=24)
all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]

# Create and fit model
model = EMModel(
    all_data=all_data,
    fit_func=rw1a1b_fit,
    param_names=["beta", "alpha"],
    param_xform=[norm2beta, norm2alpha],  # Parameter transformation functions
)

# Fit the model
result = model.fit(verbose=1)
print(f"Convergence: {result.convergence}")

# Access results
output_dict = model.calculate_final_arrays() 
estimated_params = output_dict['params']  # Shape: (n_subjects, n_params)
print(f"Estimated parameters shape: {estimated_params.shape}")

# Get output dict (model predictions, etc.) --> defined from your fit function when `out="all"`
arrays = model.calculate_final_arrays()
print(f"Available arrays: {list(arrays.keys())}")

# Plot recovery
for param_idx, param_label in enumerate(['beta','alpha']):
    simulated_param = sim['params'][:,param_idx]
    fitted_params = output_dict['params'][:,param_idx]
    ax = plotting.plot_scatter(simulated_param, f'Simulated {param_label}', 
                 fitted_params, f'Estimated {param_label}')
```

### Parameter Recovery Analysis

We can also use the `EMModel.recover()` method to perform parameter recovery directly if you provide the 

```python
# Create model object
model = EMModel(
    all_data=all_data,
    fit_func=rw1a1b_fit,
    param_names=["beta", "alpha"],
    param_xform=[norm2beta, norm2alpha],  # Parameter transformation functions
    simulate_func=rw1a1b_simulate
)

# Perform parameter recovery
recovery_dict = model.recover(
    true_params=true_params,
    nblocks=3, ntrials=24 # settings for simulate function
)

# Plot recovery results
fig = model.plot_recovery(recovery_dict, figsize=(10, 4))
```

### Model Comparison

When we have two different models, we can use the `ModelComparison` class to compare them using various metrics. The package provides several metrics for model comparison:

* **LME** (Log Model Evidence): Laplace approximation to marginal likelihood
* **BIC** (Bayesian Information Criterion): Penalizes model complexity
* **Integrated BIC**: Monte Carlo approximation accounting for parameter uncertainty

```python
from pyem.api import EMModel
from pyem.core.compare import ModelComparison
from pyem.models.rl import rw1a1b_fit, rw1a1b_simulate, rw2a1b_fit, rw2a1b_simulate
from pyem.utils.math import norm2alpha, norm2beta

# Create multiple models for comparison
model1 = EMModel(all_data, rw1a1b_fit, 
                 param_names=["beta", "alpha"],
                 param_xform=[norm2beta, norm2alpha],
                 simulate_func=rw1a1b_simulate)

model2 = EMModel(all_data, rw2a1b_fit,
                 param_names=["beta", "alpha_pos", "alpha_neg"],
                 param_xform=[norm2beta, norm2alpha, norm2alpha],
                 simulate_func=rw2a1b_simulate)

# Fit both models
model1.fit(verbose=0)
model2.fit(verbose=0)

# Compare models
mc = ModelComparison([model1, model2], ["RW1", "RW2"])
bicint_kwargs = {"nsamples":2000, "func_output":"all", "nll_key":"nll"}
r2_kwargs = {"ntrials": nblocks*ntrials, "nopts": 2} # two-armed bandit has 2 options
compare_results = mc.compare(bicint_kwargs=bicint_kwargs, r2_kwargs=r2_kwargs)

# Print comparison results
for result in compare_results:
    print(
        f"{result.name}: "
        f"LME = {result.LME:.2f}, "
        f"BICint = {result.BICint:.2f}, "
        f"R^2 = {result.R2:.2f}"
    )
```

We can also compute these metrics using the EMModel class directly.

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
from pyem.api import EMModel
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
mc = ModelComparison([model1, model2], ["RW1", "RW2"])
df = mc.identify(
    rounds=10,
    sim_kwargs={"nblocks":3, "ntrials": 24}, # args for simulate_func
    fit_kwargs={"mstep_maxit": 50},
    r2_kwargs={"ntrials": 3*24, "nopts": 2},
    verbose=1,
)

print(df.head())
#    Simulated  Estimated  LME  BICint  bestlme  bestbic
# 0  RW1        RW1 ...
# 1  RW1        RW2 ...
# 2  RW2        RW1 ...
# 3  RW2        RW2 ...

# Plot results as proportion of rounds won
mc.plot_identifiability(df, metric="LME")
mc.plot_identifiability(df, metric="BICint")
```

### Miscellanous functions

Many computational models have bounded parameters (e.g., learning rates between 0-1). The package uses transformation functions to map between:

* **Normalized space**: Unbounded parameters used during optimization
* **Parameter space**: Bounded parameters used in model computations

Example:
```python
# Learning rate: 0 ≤ α ≤ 1
alpha_normalized = 0.5  # Unbounded
alpha_natural = norm2alpha(alpha_normalized)  # Bounded to [0,1]
```

If you provide function to transform parameters from Gaussian to parameter space (see Daw, 2011), then the following can be used to access them from the EMModel class.

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
def my_model_fit(params, choices, rewards, *, prior=None, output="npl"):
    """
    Fit function for your custom model.
    
    Args:
        params: Parameter values in normalized space
        choices: Choice data for one subject (CAN BE ANYTHING)
        rewards: Reward data for one subject (CAN BE ANYTHING)
        prior: Prior distribution (used by EM algorithm)
        output: "npl" for negative posterior likelihood, "nll" for negative log-likelihood, "all" for full output
        
    Returns:
        Negative posterior likelihood (float) or full dictionary if output="all"
    """
    # Transform parameters from normalized to parameter space
    alpha = norm2alpha(params[0])
    beta = norm2beta(params[1])
    
    # Bounds checking
    if not (0 <= alpha <= 1): return 1e7
    if not (0.001 <= beta <= 20): return 1e7
    
    # Your model implementation here
    nll = ...
    
    if output == "nll":
        return nll
    elif output == "all":
        return {"params": [alpha, beta], "choices": choices, "rewards": rewards, "nll": nll}
    
    # Compute negative posterior likelihood
    if prior is not None:
        nlp = -prior.logpdf(np.asarray(params))
        return nll + nlp
    return nll

def my_model_simulate(params, **kwargs):
    """
    Simulation function for your custom model.
    
    Args:
        params: Parameter values in parameter space (n_subjects x n_params)
        **kwargs: Additional simulation parameters
        
    Returns:
        Dictionary with keys (CAN BE ANYTHING + "nll"): "params", "choices", "rewards", "nll", etc.
    """

    # Your simulation implementation here
    ...

    return {"params": params, "choices": choices, "rewards": rewards, "nll": nll}
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
    def calculate_final_arrays(self) -> dict
```

### ModelComparison Class

Class for comparing the performance of different models:

```python
class ModelComparison:
    def __init__(self, models, names)
    def compare(self) -> list
    def indentify(self) -> pd.DataFrame
    def plot_identifiability(self) -> plt.Figure
```

### Utility Functions

* **Parameter transformations** (`pyem.utils.math`): `norm2alpha()`, `norm2beta()`, `alpha2norm()`, `beta2norm()`
* **Statistics** (`pyem.utils.stats`): `calc_BICint()`, `calc_LME()`, `pseudo_r2_from_nll()`
* **Plotting** (`pyem.utils.plotting`): Various visualization functions

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

* Python >= 3.8
* numpy >= 1.22
* scipy >= 1.10
* pandas >= 1.5
* matplotlib >= 3.5
* joblib >= 1.3

## Examples

See the `examples/` directory for detailed tutorials:

* `examples/RW.ipynb`: Rescorla-Wagner model parameter recovery
* `examples/EMClass.ipynb`: Using the EMModel class interface

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
3. **MAP estimation**: Incorporates prior beliefs to regularize parameter estimates

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