<div align="center">

<a target="_blank" rel="noopener noreferrer" href="https://doi.org/10.5281/zenodo.10415396">![DOI:10.5281/zenodo.10415396](https://zenodo.org/badge/DOI/10.5281/zenodo.10415396.svg)</a> <a target="_blank" rel="noopener noreferrer" href="https://github.com/shawnrhoads/pyEM">![GitHub last update](https://img.shields.io/github/last-commit/shawnrhoads/pyEM?color=blue&label=last%20update)</a> <a target="_blank" rel="noopener noreferrer" href="https://www.buymeacoffee.com/shawnrhoads">![BuyMeACoffee](https://img.shields.io/static/v1?message=contribute%20caffeine&label=%20&style=square&logo=Buy%20Me%20A%20Coffee&labelColor=5c5c5c&color=lightgrey)</a>

# pyEM: Expectation Maximization with MAP estimation in Python

</div>

<b>If you use these materials for teaching or research, please use the following citation:</b>
> Rhoads, S. A. (2023). pyEM: Expectation Maximization with MAP estimation in Python. Zenodo. <a href="https://doi.org/10.5281/zenodo.10415396">https://doi.org/10.5281/zenodo.10415396</a>

This is a Python implementation of the Hierarchical Expectation Maximization algorithm with MAP estimation for fitting computational models to behavioral data. The package provides both low-level fitting functions and a high-level, sklearn-like interface for ease of use.

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
from pyem.api import EMModel
from pyem.models.rl import rw1a1b_simulate, rw1a1b_fit
from pyem.utils.math import norm2beta, norm2alpha

# Generate sample data
true_params = np.array([[0.5, 0.3], [0.7, 0.4], [0.2, 0.6]])  # beta, alpha for 3 subjects
sim = rw1a1b_simulate(true_params, nblocks=3, ntrials=24)
all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]

# Create and fit model
model = EMModel(
    all_data=all_data,
    fit_func=rw1a1b_fit,
    param_names=["beta", "alpha"],
    param_xform=[norm2beta, norm2alpha],  # Parameter transformation functions
    simulate_func=rw1a1b_simulate
)

# Fit the model
result = model.fit(verbose=1, mstep_maxit=100)

# Access results
estimated_params = result.m.T  # Shape: (n_subjects, n_params)
posterior_mu = result.posterior_mu
posterior_sigma = result.posterior_sigma

print(f"Estimated parameters shape: {estimated_params.shape}")
print(f"Convergence: {result.convergence}")
```

### Parameter Recovery Analysis

```python
# Perform parameter recovery
recovery_dict = model.recover(
    true_params=true_params,
    nblocks=3,
    ntrials=24
)

# Plot recovery results
fig = model.plot_recovery(recovery_dict, figsize=(10, 4))

# Print recovery metrics
print(f"Overall correlation: {recovery_dict['correlation']:.3f}")
print(f"RMSE: {recovery_dict['rmse']:.3f}")
print(f"MAE: {recovery_dict['mae']:.3f}")
```

### Model Comparison

```python
from pyem.core.compare import ModelComparison

# Create multiple models for comparison
model1 = EMModel(all_data, rw1a1b_fit, ["beta", "alpha"])
model2 = EMModel(all_data, rw2a1b_fit, ["beta", "alpha_pos", "alpha_neg"])

# Fit both models
model1.fit(verbose=0)
model2.fit(verbose=0)

# Compare models
comparison = ModelComparison([model1, model2], ["RW1", "RW2"])
results = comparison.compare()

# Print comparison results
for result in results:
    print(f"{result.name}: LME = {result.LME:.2f}")
```

### Advanced Analysis

```python
# Compute integrated BIC
bicint = model.compute_integrated_bic(nsamples=1000)

# Compute Laplace approximation for LME
lap, lme, good = model.compute_lme()

# Calculate final arrays (model predictions, etc.)
arrays = model.calculate_final_arrays()
print(f"Available arrays: {list(arrays.keys())}")

# Access parameter transformations
beta_transform = model.get_param_transform("beta")
alpha_transform = model.get_param_transform("alpha")

# Transform parameters
transformed_beta = beta_transform(0.5)  # Convert from normalized to natural space
transformed_alpha = alpha_transform(0.3)
```

## Available Models

The package includes several pre-implemented models:

### Reinforcement Learning Models (`pyem.models.rl`)

* **`rw1a1b_simulate/fit`**: Rescorla-Wagner model with single learning rate
* **`rw2a1b_simulate/fit`**: Rescorla-Wagner model with separate learning rates for positive/negative prediction errors

### Bayesian Models (`pyem.models.bayes`)

* **`simulate/fit`**: Bayesian learning model for categorization tasks

### Creating Custom Models

To create a custom model, implement two functions:

```python
def my_model_fit(params, choices, rewards, prior=None, output="npl"):
    """
    Fit function for your custom model.
    
    Args:
        params: Parameter values in normalized space
        choices: Choice data for one subject
        rewards: Reward data for one subject  
        prior: Prior distribution (used by EM algorithm)
        output: "npl" for negative posterior likelihood, "nll" for negative log-likelihood, "all" for full output
        
    Returns:
        Negative posterior likelihood (float) or full dictionary if output="all"
    """
    # Transform parameters from normalized to natural space
    alpha = norm2alpha(params[0])
    beta = norm2beta(params[1])
    
    # Bounds checking
    if not (0 <= alpha <= 1): return 1e7
    if not (0.001 <= beta <= 20): return 1e7
    
    # Your model implementation here
    nll = compute_negative_log_likelihood(alpha, beta, choices, rewards)
    
    if output == "nll":
        return nll
    elif output == "all":
        return {"params": [alpha, beta], "nll": nll, "predictions": predictions}
    
    # Compute negative posterior likelihood
    if prior is not None:
        nlp = -prior.logpdf(np.asarray(params))
        return nll + nlp
    return nll

def my_model_simulate(params, **kwargs):
    """
    Simulation function for your custom model.
    
    Args:
        params: Parameter values in natural space (n_subjects x n_params)
        **kwargs: Additional simulation parameters
        
    Returns:
        Dictionary with keys: "params", "choices", "rewards", "nll", etc.
    """
    # Your simulation implementation here
    return {"params": params, "choices": choices, "rewards": rewards, "nll": nll}
```

## Key Classes and Functions

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

For comparing multiple models:

```python
class ModelComparison:
    def __init__(self, models, names)
    def compare(self) -> list
    def identifiability_analysis(self, **kwargs)
    def plot_identifiability(self, **kwargs) -> plt.Figure
```

### Utility Functions

* **Parameter transformations** (`pyem.utils.math`): `norm2alpha()`, `norm2beta()`, `alpha2norm()`, `beta2norm()`
* **Statistics** (`pyem.utils.stats`): `calc_BICint()`, `calc_LME()`, `pseudo_r2_from_nll()`
* **Plotting** (`pyem.utils.plotting`): Various visualization functions

## Installation

### From GitHub (recommended)

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

## Key Concepts

### Hierarchical EM with MAP Estimation

The algorithm fits models using a hierarchical approach where:

1. **E-step**: Estimates subject-specific parameters given population-level priors
2. **M-step**: Updates population-level priors given subject-specific parameters
3. **MAP estimation**: Incorporates prior beliefs to regularize parameter estimates

### Parameter Transformations

Many computational models have bounded parameters (e.g., learning rates between 0-1). The package uses transformation functions to map between:

* **Normalized space**: Unbounded parameters used during optimization
* **Natural space**: Bounded parameters used in model computations

Example:
```python
# Learning rate: 0 ≤ α ≤ 1
alpha_normalized = 0.5  # Unbounded
alpha_natural = norm2alpha(alpha_normalized)  # Bounded to [0,1]
```

### Model Comparison

The package provides several metrics for model comparison:

* **LME** (Log Model Evidence): Laplace approximation to marginal likelihood
* **BIC** (Bayesian Information Criterion): Penalizes model complexity
* **Integrated BIC**: Monte Carlo approximation accounting for parameter uncertainty

## Contributing

Contributions are welcome! Please see the contributing guidelines for details on:

* Reporting bugs
* Suggesting enhancements
* Adding new models
* Improving documentation

## References

**Code originally adapted for Python from:**
> Wittmann, M. K., et al. (2020). Global reward state affects learning and activity in raphe nucleus and anterior insula in monkeys. Nature Communications, 11(1), 3771.

> Cutler, J., et al. (2021). Ageing is associated with disrupted reinforcement learning whilst learning to help others is preserved. Nature Communications, 12(1), 4440.

**See also:**
> Daw, N. D. (2011). Trial-by-trial data analysis using computational models. Decision making, affect, and learning: Attention and performance XXIII.

> Huys, Q. J., et al. (2011). Disentangling the roles of approach, activation and valence in instrumental and pavlovian responding. PLoS computational biology, 7(4), e1002028.

**For MATLAB implementations:**
* https://github.com/sjgershm/mfit
* https://github.com/mpc-ucl/emfit
* https://osf.io/s7z6j