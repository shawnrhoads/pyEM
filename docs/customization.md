# Creating Custom Models

The entire purpose of pyEM is the make it very easy to generate new models for simulation and fitting. Every model follows the same template: a pair of `_sim`/`_fit` functions plus a `ModelSpec` that bundles them with a hand-authored `id`/`description`/`spec`. To create a custom model, follow the same template:

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
    # 
    # ------------------------- 

    fval = calc_fval(nll, params, output="nll")

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
