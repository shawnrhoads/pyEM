# pyEM

<img class="pyem-logo pyem-logo--light" src="assets/pyem-logo-horizontal-editable.svg" alt="pyEM" width="420">
<img class="pyem-logo pyem-logo--dark" src="assets/pyem-logo-horizontal-dark-editable.svg" alt="pyEM" width="420">

pyEM provides Expectation Maximization with MAP in Python for fitting cognitive computational models to behavioral data, using maximum a posteriori (MAP) estimation. Rather than fitting each subject in isolation, the EM algorithm alternates between estimating each subject's parameters (E-step) and updating a group-level Gaussian prior over those parameters (M-step), which regularizes noisy individual fits toward the group. The main entry point is the [`EMModel`](api/emmodel.md) class, which wraps a subject's data, a model's fitting function, and its parameter names/transforms, and returns a `FitResult` with subject- and group-level estimates. pyEM ships with several built-in model families — reinforcement learning (model-free and model-based), GLM/regression, prospect theory, discounting, signal detection theory, Bayesian inference, and drift-diffusion models — and includes utilities for parameter recovery and model comparison (integrated BIC, log model evidence).

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [API Reference](api/index.md)
- [Examples](examples/rl_mf.ipynb)
