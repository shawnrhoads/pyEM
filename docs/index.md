#
<img class="pyem-logo pyem-logo--light" src="assets/pyem-logo-horizontal-editable.svg" alt="pyEM" width="420">
<img class="pyem-logo pyem-logo--dark" src="assets/pyem-logo-horizontal-dark-editable.svg" alt="pyEM" width="420">

pyEM provides Expectation Maximization with MAP in Python for fitting cognitive computational models to behavioral data, using maximum a posteriori (MAP) estimation. Rather than fitting each subject in isolation, the EM algorithm alternates between estimating each subject's parameters (E-step) and updating a group-level Gaussian prior over those parameters (M-step), which regularizes noisy individual fits. The [`EMModel`](api/emmodel.md) class is the main entry point for fitting a model to data. This wraps a subject's data, a model's fitting function, and its parameter names/transforms, and returns a `FitResult` with subject- and group-level estimates. 

pyEM ships with several built-in model families, which are included for **teaching** and **demonstrations** of this package's flexibility. They can also be used for model fitting in your own research, but please note that they use specific task structures that might not match your own study design or research question. I am happy to chat if you have any questions about this. If this is the case, you can easily create your own custom model by following the same template as the built-in models. pyEM also provides utilities for model comparison and parameter recovery, which can be used to evaluate the identifiability of your own models.

**Table of Contents**

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [Custom Models](customization.md)
- [API Reference](api/index.md)
- [FAQ](faq.md)
- [Troubleshooting](troubleshooting.md)

<b>If you use these materials for teaching or research, please use the following citations:</b>

> Rhoads, S. A. (2023). pyEM: Expectation Maximization with MAP estimation in Python. Zenodo. <a href="https://doi.org/10.5281/zenodo.10415396">https://doi.org/10.5281/zenodo.10415396</a>

> Rhoads, S. A., Gan, L., Berluti, K., O'Connell, K., Cutler, J., Lockwood, P. L., & Marsh, A. A. (2025). Neurocomputational basis of learning when choices simultaneously affect both oneself and others. In press at *Nature Communications*. <a href="https://doi.org/10.1038/s41467-025-64424-9">https://doi.org/10.1038/s41467-025-64424-9</a>
