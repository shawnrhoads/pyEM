# API Reference

pyEM's import surface has four layers. The top-level package exposes the
high-level fitting interface directly: `from pyem import EMModel, FitResult`.
Core building blocks (the EM loop, optimizer, model comparison, group M-step
families, priors, parameter recovery, and `ModelSpec`) live under the `core`
subpackage and are typically imported as `pyem.core.<mod>` (e.g.
`from pyem.core import em, optim, compare, priors`). Shared helpers (math
transforms, summary statistics, plotting) live under `pyem.utils.<mod>` (e.g.
`from pyem.utils.math import norm2beta, norm2alpha`). Individual computational
models are imported from their model module under `pyem.models`, e.g.
`from pyem.models.rl_mf import rw1a1b_model`.