# Parameter recovery notebook pattern

Use this reference to implement `examples/{model_class}.ipynb` even when base example notebooks are unavailable.

This pattern is distilled from the repository notebooks under `examples/` (`rl.ipynb`, `bayes.ipynb`, `glm.ipynb`):

- Intro markdown title + task subtitle.
- Import block (`numpy`, plotting, model sim/fit, `EMModel`).
- Simulation setup cell.
- Simulation execution cell.
- Fit-and-recover cell using `EMModel.recover(...)`.
- Parameter recovery scatter plots with identity lines.

## Required sections

1. Model and task overview.
2. Parameter specification (true generating parameters).
3. Simulation run.
4. Fit simulated behavior.
5. Parameter recovery plot.
6. Brief interpretation.

## Template source

Use `references/example-notebook-template.json` as the base cell template. Replace all placeholders (for example `{model_name}`, `{model_class}`, bounds, and parameter names).

## Minimal recovery workflow

1. Choose `N` synthetic subjects (e.g., `N=50`).
2. Sample true parameters in natural space.
3. Run `{model_name}_sim` to generate behavior.
4. Fit each synthetic subject with `{model_name}_fit` via `EMModel.recover`.
5. Compare true vs recovered parameters.

## Plot requirements

- One subplot per parameter.
- X-axis: true values.
- Y-axis: recovered values.
- Add identity line `y=x`.
- Report Pearson correlation `r` in each panel title.

## Minimal plotting snippet

```python
fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
for i, ax in enumerate(np.atleast_1d(axes)):
    ax.scatter(true_params[:, i], recovered_params[:, i], alpha=0.7)
    lo = min(true_params[:, i].min(), recovered_params[:, i].min())
    hi = max(true_params[:, i].max(), recovered_params[:, i].max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    r = np.corrcoef(true_params[:, i], recovered_params[:, i])[0, 1]
    ax.set_title(f"{param_names[i]} (r={r:.2f})")
    ax.set_xlabel("True")
    ax.set_ylabel("Recovered")
plt.tight_layout()
```
