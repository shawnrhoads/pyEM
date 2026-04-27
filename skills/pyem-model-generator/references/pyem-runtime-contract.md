# Runtime contract for generated models

Generated model files should import math helpers directly from pyem:

```python
from pyem.utils.math import norm2alpha, norm2beta, softmax, calc_fval
```

The shared `modclass_utils.py` file should **not** define these math helpers.
It should only provide:

- `_alloc_sim`
- `_alloc_fit`
- `ModelSpec`
- `ParamDef`
- `spec_to_id`
- `build_params`
- `PARAM_REGISTRY`

## Function contracts

## `mod_params(nsubj, rng=None)`

- Returns `(param_names, param_xform, true_params)`.
- `true_params` shape: `(nsubj, nparams)`.

## `mod_sim(params, ..., **kwargs)`

- Returns a dictionary with stable keys appropriate to the model class.
- Common required outputs are `params` and `choices`; include `rewards` when the task/model uses reward feedback or when downstream fitting/diagnostics require it.
- Model-specific latent/diagnostic arrays such as `state`, `ev`, `pe`, and similar traces may be included, but no single latent key is required for all models.
- Uses natural-space parameters for simulation unless otherwise specified.

## `mod_fit(params, ..., prior=None, output="npl")`

- Must support `output="npl"`, `"nll"`, and optionally `"all"`.
- Uses transformed parameters (`norm2alpha`, `norm2beta`) when constraints require.
- Returns large penalty (commonly `1e7`) for invalid parameter regions.
- Uses `calc_fval` for scalar objective outputs.

## Prior handling

- Prior can be `None` or a dictionary accepted by `calc_fval`.
- Pass prior through unchanged to `calc_fval`.
