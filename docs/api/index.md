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

Each page auto-documents every public symbol from its docstring and adds a
runnable snippet per model / per function-group.

- [`emmodel.md`](emmodel.md) — EMModel & FitResult
- [`core.md`](core.md) — Core: EM, optimizer, comparison, group M-step families, priors, recovery, ModelSpec
- [`utils.md`](utils.md) — Utilities: math, stats, plotting
- [`models/rl.md`](models/rl.md) — Reinforcement Learning
- [`models/glm.md`](models/glm.md) — GLM / regression
- [`models/choice.md`](models/choice.md) — Prospect Theory & Discounting
- [`models/other.md`](models/other.md) — SDT, Bayes, DDM
