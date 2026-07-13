<img class="pyem-logo pyem-logo--light" src="assets/pyem-logo-horizontal-editable.svg" alt="pyEM" width="420">
<img class="pyem-logo pyem-logo--dark" src="assets/pyem-logo-horizontal-dark-editable.svg" alt="pyEM" width="420">

pyEM provides Expectation Maximization with MAP in Python for fitting cognitive computational models to behavioral data, using maximum a posteriori (MAP) estimation. Rather than fitting each subject in isolation, the EM algorithm alternates between estimating each subject's parameters (E-step) and updating a group-level Gaussian prior over those parameters (M-step), which regularizes noisy individual fits. The [`EMModel`](api/emmodel.md) class is the main entry point for fitting a model to data. This wraps a subject's data, a model's fitting function, and its parameter names/transforms, and returns a `FitResult` with subject- and group-level estimates. 

pyEM ships with several built-in model families, which are included for **teaching** and **demonstrations** of this package's flexibility. They can also be used for model fitting in your own research, but please note that they use specific task structures that might not match your own study design or research question. I am happy to chat if you have any questions about this. If this is the case, you can easily create your own custom model by following the same template as the built-in models. pyEM also provides utilities for model comparison and parameter recovery, which can be used to evaluate the identifiability of your own models.

**Table of Contents**
- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [Custom Models](customization.md)
- [FAQ](faq.md)
- [Troubleshooting](troubleshooting.md)
- [API Reference](api/index.md)
- [References](references.md)
