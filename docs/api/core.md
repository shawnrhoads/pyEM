# Core

The `pyem.core` subpackage implements the hierarchical Expectation-Maximization
(EM) machinery that `pyem.api.EMModel` wraps: the outer EM loop, the
subject-level (E-step) optimizer, the group-level (M-step) distribution
families, prior objects, model-comparison utilities, parameter-recovery
diagnostics, and the `ModelSpec` bundle used to register a model's identity
and entry points. Most users will interact with these through `EMModel`
rather than calling them directly, but they are documented here for anyone
building custom fitting pipelines or new group-distribution families.

## Hierarchical EM loop

`EMfit` is the low-level hierarchical-EM routine: it alternates a parallel
per-subject E-step (`single_subject_minimize`, MAP estimation against the
current group prior) with a group-level M-step (see
[Group M-step families](#group-m-step-families)) until a convergence
criterion is met, returning subject-level estimates, inverse-Hessians, the
final group posterior, and per-subject (negative) log-likelihood terms.
`EMConfig` bundles the loop's tunables — iteration cap, convergence method
and criterion, parallelism, RNG seed, and which M-step family to use.

::: pyem.core.em.EMfit
::: pyem.core.em.EMConfig

## Subject-level optimizer

`single_subject_minimize` performs the E-step for a single subject: it
minimizes the negative log-posterior (`objfunc(...) - logprior`) over that
subject's data with `scipy.optimize.minimize`, retrying from random restarts
until a successful optimization is found or the restart budget
(`OptimConfig.max_restarts`) is exhausted. `OptimConfig` holds the optimizer
method, solver options, restart budget, and the scale of the random restart
initializations.

::: pyem.core.optim.OptimConfig
::: pyem.core.optim.single_subject_minimize

## Model comparison

`ModelComparison` wraps one or more fitted `EMModel` instances and computes
per-model fit metrics — log model evidence (LME), integrated BIC (BICint),
and pseudo-R² — via `compare()`, and can additionally run a full
simulate-and-refit identifiability analysis across models via `identify()`
(with `plot_identifiability()` for visualizing the resulting confusion
matrix). `compare_models` is the underlying function that produces the list
of `ComparisonRow` results that `ModelComparison.compare()` turns into a
DataFrame.

```python
import numpy as np
from pyem import EMModel
from pyem.core.compare import ModelComparison
from pyem.models.rl_mf import rw1a1b_sim, rw1a1b_fit
from pyem.utils.math import norm2beta, norm2alpha

rng = np.random.default_rng(0)
p1 = np.column_stack([norm2beta(rng.normal(size=15)), norm2alpha(rng.normal(size=15))])
sim = rw1a1b_sim(p1, nblocks=2, ntrials=20, seed=0)
data = [[sim["choices"][s], sim["rewards"][s]] for s in range(15)]
m1 = EMModel(data, rw1a1b_fit, ["beta", "alpha"], param_xform=[norm2beta, norm2alpha], simulate_func=rw1a1b_sim)
m1.fit(verbose=0)
cmp = ModelComparison([m1], ["rw1a1b"])
print(cmp.compare(bicint_kwargs={"nsamples": 50, "func_output": "all", "nll_key": "nll"}))
```

::: pyem.core.compare.compare_models
::: pyem.core.compare.ComparisonRow
::: pyem.core.compare.ModelComparison

## Group M-step families

Each group-distribution family (Gaussian, Laplace, Student-t, Cauchy)
implements the same three-method interface — `update(m, inv_h)` to run the
empirical-Bayes M-step over subject-level MAP means and inverse-Hessians,
`moments(hyper)` to return the group mean/variance and a validity flag, and
`make_prior(hyper)` to build the `Prior` object fed back into the next
E-step — so `EMfit` can swap between them via the `EMConfig.mstep` name.
`make_group` is the factory that dispatches a family name (and, for
Student-t, a degrees-of-freedom `df`) to the corresponding class.

```python
import numpy as np
from pyem.core.groupdist import make_group
g = make_group("student_t", df=8.0)
m = np.random.default_rng(0).normal(size=(2, 20))
inv_h = np.stack([np.eye(2) for _ in range(20)], axis=-1)
print(g.moments(g.update(m, inv_h)))
```

::: pyem.core.groupdist.make_group
::: pyem.core.groupdist.GaussianGroup
::: pyem.core.groupdist.LaplaceGroup
::: pyem.core.groupdist.StudentTGroup
::: pyem.core.groupdist.CauchyGroup

## Priors

`Prior` is the minimal protocol (a `logpdf(x)` method) that every prior
object satisfies; `GaussianPrior` is the default and is used throughout the
EM loop as the per-iteration regularizer built from the group posterior.
**Note:** `GaussianPrior.sigma` holds the per-parameter **variance**
(sigma², not the standard deviation) — this matches what the empirical-Bayes
group M-step reports and what `EMModel.fit(prior_sigma=...)` expects, so use
the `.variance` property alias when you want the intent to read explicitly.
`default_prior` builds a broad, weakly-informative `GaussianPrior` (variance
100) for a given number of parameters. `UniformPrior`, `LaplacePrior`,
`StudentTPrior`, and `CauchyPrior` provide alternative independent priors,
and `IndependentPrior` composes a list of 1-D priors (one per parameter)
into a single multi-parameter prior.

```python
import numpy as np
from pyem.core.priors import GaussianPrior, UniformPrior, default_prior
g = GaussianPrior(mu=[0.0, 0.0], sigma=[1.0, 4.0])   # sigma == variance
print(g.logpdf([0.0, 0.0]), g.variance)
print(default_prior(2).sigma)                          # broad default (variance 100)
print(UniformPrior(lo=[-1, 0], hi=[1, 1]).logpdf([0.0, 0.5]))
```

::: pyem.core.priors.Prior
::: pyem.core.priors.GaussianPrior
::: pyem.core.priors.default_prior
::: pyem.core.priors.UniformPrior
::: pyem.core.priors.LaplacePrior
::: pyem.core.priors.StudentTPrior
::: pyem.core.priors.CauchyPrior
::: pyem.core.priors.IndependentPrior

## Parameter recovery

`parameter_recovery` compares ground-truth subject-level parameters against
their EM-estimated counterparts, returning per-parameter Pearson correlation
and RMSE (plus the estimates themselves) bundled in a `RecoveryResult`
dataclass — the standard diagnostic for validating that a model's parameters
are identifiable from simulated data.

::: pyem.core.posterior.parameter_recovery
::: pyem.core.posterior.RecoveryResult

## Model specification

`ModelSpec` is a small, self-describing bundle (id, spec dict, description,
and the model's `params`/`sim`/`fit` callables) used to register a model's
identity and entry points in one place. It is purely additive — neither
`EMModel` nor `ModelComparison` requires a `ModelSpec` to function — but
model modules use it to expose a consistent, discoverable interface.

::: pyem.core.modelspec.ModelSpec
