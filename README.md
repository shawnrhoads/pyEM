<div align="center">

<a target="_blank" rel="noopener noreferrer" href="https://doi.org/10.5281/zenodo.10415396">![DOI:10.5281/zenodo.10415396](https://zenodo.org/badge/DOI/10.5281/zenodo.10415396.svg)</a> <a target="_blank" rel="noopener noreferrer" href="https://github.com/shawnrhoads/pyEM">![GitHub last update](https://img.shields.io/github/last-commit/shawnrhoads/pyEM?color=blue&label=last%20update)</a> <a target="_blank" rel="noopener noreferrer" href="https://www.buymeacoffee.com/shawnrhoads">![BuyMeACoffee](https://img.shields.io/static/v1?message=contribute%20caffeine&label=%20&style=square&logo=Buy%20Me%20A%20Coffee&labelColor=5c5c5c&color=lightgrey)</a>

# pyEM: Expectation Maximization with MAP estimation in Python

</div>

<b>If you use these materials for teaching or research, please use the following citation:</b>
> Rhoads, S. A. (2023). pyEM: Expectation Maximization with MAP estimation in Python. Zenodo. <a href="https://doi.org/10.5281/zenodo.10415396">https://doi.org/10.5281/zenodo.10415396</a>

This is a Python implementation of the Hierarchical Expectation Maximization algorithm with MAP estimation for fitting models to behavioral data. [See below](#key-concepts) for more information on the algorithm.

## Relevant Modules
* `pyEM.fitting`: contains the main function (`EMfit`) for fitting models
* `pyEM.math`: contains functions used for fitting
* `pyEM.plotting`: contains functions for simple plotting

## Usage
Users should create new functions based on their modeling needs:
1. Simulation function to simulate behavior (see `examples.rw_models.simulate`)
2. Fit function to fit the model to the behavior (see `examples.rw_models.fit`)

You will use `EMfit()` to fit the model to the data. This function takes the following inputs:
- all behavioral data: a list of lists (level 1: subject, level 2: relevant `np.array` containing behavior for each subject) or a list of pd.DataFrames (level 1: subject, level 2: relevant `pd.DataFrame` containing behavior for each subject)
- objective function (e.g., `examples.rw_models.fit`): a function that takes level 2 inputs from above: (e.g., single `np.array` or `pd.DataFrame` for a single subject) and outputs the negative posterior likelihood based on likelihood and prior probabilities (see explanation of implementation below)
- parameter names: a list of parameter names in the correct order (e.g., `['alpha', 'beta']`)

### Setting Up Your Objective Function
Your objective function `fit()` function is that main implementation of your custom model. It should take the following inputs:
- `params` (list): list of parameter estimates
- `behavioral_data`: this can be a `np.array` or `pd.DataFrame` (choose whichever is more convenient for your model)
- `prior=None`: this is for the EM algorithm, and should have a default value as `None` (see below for more information)
- `output='npl'` (str): this is also for the EM algorithm, and should have a default value as `'npl'` (see below for more information)

You can also add additional inputs as needed.

At the top of your function, please include code to convert relevant parameters from Gaussian space to parameter space (see Daw et al., 2011 below). For example, if you have a parameter `lr` that is bounded between 0 and 1, you can convert it from Gaussian space to parameter space using the following code (you can add custom functions if needed):
```python
lr = norm2alpha(params[0])
```

After your transformation, please also ensure that your parameters are in the correct range. For example, if you have parameters `lr` (bounded between 0 and 1) and `inv_tmp` (bounded between 0.00001 and 10), you can add the following code to ensure that the parameter is in the correct range. Here, we are returning a very large number if the parameter is out of range, which will effectively prevent the algorithm from using that parameter value:
```python
this_alpha_bounds = [0, 1]
if lr < min(this_alpha_bounds) or lr > max(this_alpha_bounds):
    return 10000000

this_beta_bounds = [0.00001, 10]
if inv_tmp < min(this_beta_bounds) or inv_tmp > max(this_beta_bounds):
    return 10000000
```

At the bottom of your function, please return the negative posterior likelihood. This is the negative log-likelihood (e.g., from your choice likelihoods) multiplied by the prior probability. You can use the following code to return the negative posterior likelihood:
```python
# you can compute your negative log likelihood first (this might change depending on your model)
# here, we are assuming you have a list or np.array of choice likelihoods
negll = -np.log(np.nansum(choice_likelihoods))

# then compute the negative posterior likelihood
# you can copy and paste the following code (assuming that your negative log likelihood is `negll`)
if output == 'npl':
    if prior is not None: # EM-fit: P(Choices | h) * P(h | O)
        fval = -(-negll + prior['logpdf'](params))       
        if np.isinf(fval):
            fval = 10000000
        return fval
    else: # NLL fit 
        return negll
```

## Requirements
This algorithm requires Python 3.7.10 with the following packages:
```
numpy: 1.21.6
scipy: 1.6.2
joblib: 1.1.0
matplotlib: 3.5.3
seaborn: 0.12.2
pandas: 1.1.5
tqdm: 4.65.0
```

We also use:
```
copy
datetime
pickle
sys
``` 

## Installation
To install the package, you can use `pip install` in a new Anaconda environment (recommended), but you can also just `pip install` it into your current environment:
```
conda create --name emfit pip python=3.7.10
conda activate emfit
python -m pip install git+https://github.com/shawnrhoads/pyEM.git
```

To update the package, you can use pip:
```
python -m pip install --upgrade git+https://github.com/shawnrhoads/pyEM.git
```

## Examples
See `examples/RW.ipynb` with an example notebook on implementing the algorithm using the Rescorla-Wagner model of reinforcement learning. This notebook simulates behavior and fits the model to the simulated data to demonstrate how hierarchical EM-MAP can be used for parameter recovery.

## Future Implementations
In future versions, I would love to add support using Python classes. For exampe, there could be a base model class with `simulate()` and `fit()` methods that can be inherited by other models. This would allow for a more flexible implementation of the algorithm, and would allow for the use of different models without having to change the code.

## For Contributors
This is meant to be a basic implementation of hierarchical EM with MAP. There is still so much left out. That being said, other researchers and educators are invited to help improve and expand the code here!

Here are some ways you can help!
- If you spot an error (e.g., typo, bug, inaccurate descriptions, etc.), please open a new issue on GitHub by clicking on the GitHub Icon in the top right corner on any page and selecting "open issue". Alternatively, you can <a target="_blank" rel="noopener noreferrer" href="https://github.com/shawnrhoads/pyEM/issues/new?labels=bug&template=issue-template.yml">open a new issue</a> directly through GitHub.
- If there is inadvertently omitted credit for any content that was generated by others, please also <a target="_blank" rel="noopener noreferrer" href="https://github.com/shawnrhoads/pyEM/issues/new?labels=enhancement&template=enhancement-template.yml">open a new issue</a> directly through GitHub.
- If you have an idea for a new example tutorial or a new module to include, please either <a target="_blank" rel="noopener noreferrer" href="https://github.com/shawnrhoads/pyEM/issues/new?labels=enhancement&template=enhancement-template.yml">open a new issue</a> and/or submit a pull request directly to the repository on GitHub.

<hr>

## Key Concepts
*Negative Log-Likelihood* 

The negative log-likelihood is a measure of how well the model fits the observed data. It is obtained by taking the negative natural logarithm of the likelihood function. The goal of MLE is to find the parameter values that minimize the negative log-likelihood, effectively maximizing the likelihood of the observed data given the model.

*Prior Probability*

The prior probability represents your knowledge or belief about the parameters before observing the data. It is typically based on some prior information or assumptions. In this case, you are using a normal distribution to represent your prior beliefs about the parameters, with mean $\mu$ and standard deviation $\sqrt{\sigma}$.

*MAP Estimation*

In MAP estimation, you are incorporating the prior probability into the estimation process. Instead of only maximizing the likelihood (as in MLE), you are maximizing the posterior probability, which combines the likelihood and the prior. Mathematically, MAP estimation can be expressed as: 

$argmax_{\theta} (likelihood(\theta | data) * prior(\theta))$

where $\theta$ represents the model parameters

We are effectively combining the likelihood and the prior in a way that biases the parameter estimation towards the prior beliefs. Since we are maximizing this combined term, we are seeking parameter values that not only fit the data well (as indicated by the likelihood) but also align with the prior probability distribution.

**Code originally adapted for Python from:**
<blockquote>Wittmann, M. K., Fouragnan, E., Folloni, D., Klein-Flügge, M. C., Chau, B. K., Khamassi, M., & Rushworth, M. F. (2020). Global reward state affects learning and activity in raphe nucleus and anterior insula in monkeys. Nature Communications, 11(1), 3771. https://doi.org/10.1038/s41467-020-17343-w</blockquote>

See also:
<blockquote>Daw, N. D. (2011). Trial-by-trial data analysis using computational models. Decision making, affect, and learning: Attention and performance XXIII, 23(1). https://doi.org/10.1093/acprof:oso/9780199600434.003.0001 [<a href="https://www.princeton.edu/~ndaw/d10.pdf">pdf</a>]</blockquote>

<blockquote>Huys, Q. J., Cools, R., Gölzer, M., Friedel, E., Heinz, A., Dolan, R. J., & Dayan, P. (2011). Disentangling the roles of approach, activation and valence in instrumental and pavlovian responding. PLoS computational biology, 7(4), e1002028. https://doi.org/10.1371/journal.pcbi.1002028 </blockquote>

**For MATLAB flavors of this algorithm:**
- https://github.com/sjgershm/mfit
- https://github.com/mpc-ucl/emfit
- https://osf.io/s7z6j