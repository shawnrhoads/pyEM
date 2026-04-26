# pyEM runtime contract (offline reference)

Use this file when the full `pyem` package is unavailable. It defines minimal contracts needed to generate compatible model modules.

## Expected utility imports

Preferred import in generated model files:

```python
from ..utils.math import softmax, norm2alpha, norm2beta, calc_fval
```

## Utility behavior

### `softmax(values, beta)`

- Inputs:
  - `values`: 1D array-like action values.
  - `beta`: inverse temperature (`> 0`).
- Output:
  - Probability vector matching `values` length.
- Stable form:

```python
z = beta * (values - np.max(values))
exp_z = np.exp(z)
p = exp_z / np.sum(exp_z)
```

### `norm2alpha(x)`

- Maps unconstrained real `x` to `(0, 1)`.
- Logistic form is acceptable:

```python
alpha = 1.0 / (1.0 + np.exp(-x))
```

### `norm2beta(x)`

- Maps unconstrained real `x` to `(1e-5, 20]`.
- Compatible bounded-sigmoid form:

```python
beta = 1e-5 + (20.0 - 1e-5) / (1.0 + np.exp(-x))
```

### `calc_fval(nll, params, prior=None, output="npl")`

- `output="nll"`: return `nll`.
- `output="npl"`: return `nll - log_prior(params)` if prior exists; else `nll`.
- `output="all"`: typically handled by caller model function.

## Prior contract

Use a lightweight prior dictionary that can be passed through unchanged to pyEM:

```python
prior = {
    "mu": np.array([...]),
    "sigma": np.array([...]),
}
```

If prior shape mismatches params, return a large penalty value (commonly `1e7`).

## Fit function contract

- Signature pattern:
  - `{model_name}_fit(params, *, prior=None, output="npl", **kwargs)`
- Must support at least `output in {"npl", "nll", "all"}`.
- Must return scalar for `"npl"`/`"nll"`.
- For invalid transformed params, return `1e7`.
