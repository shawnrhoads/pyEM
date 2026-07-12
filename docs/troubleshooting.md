# Troubleshooting

## `TypeError` from `EMModel.recover()` when using a GLM model

**Symptom:** Calling `model.recover(true_params, pr_inputs=[...])` on a GLM family raises a `TypeError` complaining that `simulate_func` must return a dict.

**Cause:** `recover()` requires `simulate_func` to return a dict of named arrays so it can select fields by name via `pr_inputs`. The GLM family's `*_sim` functions return an `(X, Y)` tuple instead, which isn't compatible with that mechanism.

**Fix:** Don't use `recover()` for GLM models. Either fit the GLM directly with `EMModel(...).fit(...)` using the `(X, Y)` output packaged into `all_data` yourself, or write a thin wrapper around the GLM's `*_sim` that returns `{"X": X, "Y": Y}` if you specifically need the `recover()`/`plot_recovery()` workflow.

## Every subject's NLL shows up as exactly `1e7`

**Symptom:** `FitResult.NLL` (or the value returned by a `fit_func` call) is suspiciously uniform at `1e7` for some or all subjects, and the EM fit doesn't seem to be learning anything.

**Cause:** `1e7` is the capped fallback value used whenever a fit function's objective is non-finite (e.g. `NaN`/`inf` from a log of zero, or a parameter falling outside its natural bounds — most built-in `fit_func`s explicitly return `1e7` when a transformed parameter like `beta` or `alpha` is outside its valid range). This is a deliberate cap (see `calc_fval` in `pyem.utils.math`) that keeps the optimizer from crashing on invalid parameter proposals, not a real likelihood value.

**Fix:** Check that your `param_xform` list matches the bounds your `fit_func` actually enforces (e.g. `norm2beta`/`norm2alpha` for a `(0, 20)`/`(0, 1)` model), that your data doesn't contain values (choices, rewards) the fit function doesn't expect, and that your starting Gaussian-space values aren't so extreme that every transformed parameter falls outside bounds on the first EM iteration.

## `compute_lme()` returns `NaN` or seems to drop subjects

**Symptom:** `result.compute_lme()` (or the per-subject Laplace approximation) has `NaN`/zero entries, or the reported subject count in diagnostics is lower than expected.

**Cause:** The Laplace approximation used by `calc_LME` requires a positive-definite Hessian (`inv_h`) for each subject. When `np.linalg.slogdet` reports a non-positive Hessian for a subject (common for subjects whose fit sits at a bound or is otherwise poorly identified), that subject's contribution is treated as unreliable: it's marked as not "good" and its value is replaced by the mean of the well-behaved subjects (or `0.0` if no subject has a valid Hessian) rather than propagating a `NaN` into the summed LME.

**Fix:** Inspect which subjects have a non-positive-definite Hessian (a large fraction usually indicates weakly identified parameters or too little data per subject) and consider a stronger/narrower prior, fewer free parameters, or `compute_integrated_bic()` as a Hessian-free alternative for model comparison.

## `ValueError: mstep_maxit must be >= 1`

**Symptom:** Calling `model.fit(mstep_maxit=0)` (or any value less than 1) raises `ValueError: mstep_maxit must be >= 1`.

**Cause:** `mstep_maxit` controls the number of EM iterations and is validated up front; the EM loop cannot run with fewer than one iteration.

**Fix:** Pass `mstep_maxit >= 1` (the default is `200`). If you're trying to do a quick single-pass fit for debugging, use `mstep_maxit=1` rather than `0`.

## `ImportError` when calling `plot_recovery()` or other plotting helpers

**Symptom:** Calling `model.plot_recovery(...)` (or another plotting utility in `pyem.utils.plotting`) raises an `ImportError` mentioning seaborn.

**Cause:** Plotting helpers lazily import seaborn rather than depending on it unconditionally, since it's an optional dependency. If seaborn isn't installed, the import fails at call time with a clear message rather than at package import time.

**Fix:** Install the `viz` extra: `pip install -e ".[viz]"` (or `pip install seaborn` directly in an existing environment).
