# FAQ

## Why are my fitted parameters unbounded, and why don't they match the natural range of my model (e.g. a learning rate outside `[0, 1]`)?

The EM optimizer searches in **unbounded Gaussian space**, not in the natural parameter space of your model. You supply a `param_xform` list of transform functions (e.g. `norm2beta`, `norm2alpha` from `pyem.utils.math`) when constructing `EMModel`; these map each Gaussian-space value to its natural range (`norm2beta` → `(0, 20)`, `norm2alpha` → `(0, 1)`). Raw values on `result.posterior_mu`/`posterior_sigma` and anything you pull directly out of the optimizer are in Gaussian space — call `model.subject_params()` to get MAP estimates already mapped into natural space via `param_xform`.

## `GaussianPrior.sigma` looks too large compared to the standard deviation I expected — what is it?

`GaussianPrior.sigma` (and the `prior_sigma` argument to `EMModel.fit(...)`) holds the **variance**, not the standard deviation. If you want to be explicit about this in your own code, use the `.variance` alias on a `GaussianPrior` instance instead of `.sigma` — they refer to the same array, but `.variance` makes the units unambiguous at the call site.

## What does my model's `fit_func` need to return, and what is `output="all"` for?

A model's fit function has the signature `fit_func(params, *data, prior=None, output="npl")`. For `output="npl"` (negative posterior likelihood, the default used during EM) or `output="nll"` (negative log-likelihood), it must return a single scalar objective. For `output="all"`, it should instead return a diagnostics dictionary for that subject (containing at least an `"nll"` key, plus whatever per-trial quantities are useful for inspection — expected values, prediction errors, etc.). `output="all"` is what utilities like `get_outfit()`/`.outfit` and BIC/LME computations use to pull detailed per-subject fit information after the EM loop has converged.

## How do I choose which M-step distribution family to use?

Pass `mstep=` to `EMModel.fit(...)`. The built-in group-level M-step families are `"gaussian"` (default), `"laplace"`, `"student_t"`, and `"cauchy"`. `"gaussian"` is the standard empirical-Bayes choice and pairs with the variance-based `GaussianPrior`. The heavier-tailed families (`"laplace"`, `"student_t"`, `"cauchy"`) are useful when you expect outlier subjects whose individual fits shouldn't be allowed to distort the group-level estimate as strongly as a Gaussian prior would; `"cauchy"` is the most robust to outliers but, because the Cauchy distribution has no finite variance, its moment-based diagnostics fall back to a large finite value internally rather than reporting an actual variance.

## How does parameter recovery work in pyEM?

`EMModel.recover(true_params, pr_inputs, simulate_func=None, fit_kwargs=None, **sim_kwargs)` runs an end-to-end recovery check: it simulates data from `true_params` using `simulate_func` (or the model's own `simulate_func` if omitted), builds `all_data` by pulling the named `pr_inputs` keys out of the simulation output dict, re-fits a fresh `EMModel` on that simulated data, and returns a dict with `true_params`, `estimated_params`, the simulation output, the `fit_result`, and a per-parameter Pearson `correlation` (computed via `pyem.core.posterior.parameter_recovery`). Because `recover()` requires `simulate_func` to return a **dict** of named arrays, it will raise a `TypeError` for the GLM family's `*_sim` functions, which return `(X, Y)` tuples instead — fit GLMs directly with `EMModel.fit()` rather than through `recover()`. Once you have a `recovery_dict`, `model.plot_recovery(recovery_dict)` renders scatter plots of true vs. estimated parameters (requires the `viz` extra for seaborn).

## What's the difference between `compute_integrated_bic()` and `compute_lme()`?

Both are model-comparison quantities computed from a fitted `EMModel`. `compute_lme()` returns the Laplace-approximated log model evidence per subject plus its sum (via `pyem.utils.stats.calc_LME`), using the fitted Hessian (`inv_h`) and `NPL` from the `FitResult`; subjects whose Hessian isn't positive-definite are dropped from the Laplace approximation (their contribution is replaced by the mean of the well-behaved subjects) rather than allowed to produce a NaN or negative-variance artifact. `compute_integrated_bic()` instead computes an integrated BIC via Monte Carlo integration over the fitted group-level posterior, which doesn't rely on the per-subject Hessian at all and is a useful cross-check when many subjects have poorly conditioned Hessians.
