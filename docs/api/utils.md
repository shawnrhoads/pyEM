# Utilities

`pyem.utils` collects the shared helpers used throughout the fitting
pipeline: the Gaussian-space parameter transforms and objective-function
helper (`pyem.utils.math`), group-level fit metrics for model comparison
(`pyem.utils.stats`), and convenience plotting functions
(`pyem.utils.plotting`).

## Math transforms

These functions map between the unconstrained Gaussian space that the EM
loop optimizes over and the natural (bounded) parameter ranges that models
actually use. `norm2alpha`/`alpha2norm` are inverses mapping `R` to `(0, 1)`
(e.g. a learning rate), and `norm2beta`/`beta2norm` are inverses mapping `R`
to `(0, max_val)` (e.g. an inverse-temperature capped at `max_val`).
`softmax` is a numerically stabilized choice rule (subtracting the max
expected value before exponentiating) used to turn expected values into
choice probabilities. `calc_fval` builds the scalar objective minimized
during subject-level fitting — the negative log-posterior (`"npl"`, negative
log-likelihood plus negative log-prior) or just the negative log-likelihood
(`"nll"`) — and caps non-finite values at `1e7` so gradient-based optimizers
keep making progress instead of hitting `inf`/`nan`.

```python
import numpy as np
from pyem.utils.math import softmax, norm2alpha, norm2beta, alpha2norm

print(softmax(np.array([0.2, 0.8]), beta=3.0))
print(norm2alpha(0.0), norm2beta(0.0))          # 0.5, 10.0
print(alpha2norm(norm2alpha(0.7)))              # ~0.7 round-trip
```

::: pyem.utils.math.softmax
::: pyem.utils.math.norm2alpha
::: pyem.utils.math.alpha2norm
::: pyem.utils.math.norm2beta
::: pyem.utils.math.beta2norm
::: pyem.utils.math.calc_fval

## Statistics & model-fit metrics

These functions turn the raw output of an EM fit into the model-comparison
metrics used by `pyem.core.compare.ModelComparison` (and are usually called
through it rather than directly). `calc_LME` computes the Laplace-approximate
log model evidence from each subject's inverse-Hessian and negative-log-posterior,
summed across subjects and penalized for the number of group-level
parameters. `calc_BICint` computes the integrated BIC by Monte Carlo
integration over the group posterior — for each subject it draws parameter
samples from `N(mu, sigma)`, re-evaluates `fit_func` at each sample, and
combines the resulting negative log-likelihoods via a numerically stable
`logsumexp` before applying the usual `k*log(n)` complexity penalty.
`pseudo_r2_from_nll` computes a McFadden-style pseudo-R² comparing a model's
(median or mean) negative log-likelihood against the negative log-likelihood
of chance performance among `noptions` alternatives.

::: pyem.utils.stats.calc_LME
::: pyem.utils.stats.calc_BICint
::: pyem.utils.stats.pseudo_r2_from_nll

## Plotting

Convenience plotting helpers built on matplotlib and seaborn. **These
functions require the optional `seaborn` dependency** — install it with the
`viz` extra: `pip install -e ".[viz]"`. `plot_choices` plots subject-averaged
choice frequency over trials for a two-alternative task. `plot_scatter`
draws an x/y scatter plot (e.g. true vs. estimated parameters from a
parameter-recovery check) with an optional dashed x=y reference line and a
Pearson-r annotation; it draws onto (and returns) a matplotlib `Axes`
without calling `plt.show()`, so callers can compose multiple panels before
displaying or saving the figure.

```python
import matplotlib; matplotlib.use("Agg")
import numpy as np
from pyem.utils.plotting import plot_scatter

x = np.linspace(0, 1, 20)
y = x + np.random.default_rng(0).normal(scale=0.1, size=20)
ax = plot_scatter(x, "true", y, "estimated")
print(type(ax).__name__)
```

::: pyem.utils.plotting.plot_choices
::: pyem.utils.plotting.plot_scatter
