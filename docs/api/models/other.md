# SDT, Bayes & DDM Models

This page covers the remaining `pyem` model families: signal detection
theory (`pyem.models.sdt`), Bayesian belief updating (`pyem.models.bayes`),
and drift-diffusion models (`pyem.models.ddm`).

## Signal detection theory (pyem.models.sdt)

`pyem.models.sdt` implements the classic equal-variance Gaussian signal
detection theory (SDT) model of a yes/no old/new recognition-memory task.
On each trial the subject sees either a previously-studied ("old") or novel
("new") item. Internal evidence for old items is drawn from `N(+d'/2, 1)`
and for new items from `N(-d'/2, 1)`; the subject responds "old" whenever
the sampled evidence exceeds a single response criterion `c`.

Free parameters: sensitivity (`dprime`, d' >= 0) and response bias
(`criterion`, real-valued). Hit and false-alarm rates are then
`P(resp_old=1 | old) = Phi(d'/2 - c)` and `P(resp_old=1 | new) = Phi(-d'/2 -
c)`.

```python
import numpy as np
from pyem.models.sdt import sdt_sim, sdt_fit
sim = sdt_sim(np.array([[1.5, 0.0]]), ntrials=100, seed=0)
print(sdt_fit(np.array([0.0, 0.0]), sim["is_old"][0], sim["resp_old"][0], output="all")["nll"])
```

::: pyem.models.sdt.sdt_sim
::: pyem.models.sdt.sdt_fit
::: pyem.models.sdt.sdt_model

## Bayesian belief updating (pyem.models.bayes)

`pyem.models.bayes` implements a Bayesian belief-updating model of a
"fish task": on each trial a new coloured fish appears, and the subject
must infer which of three ponds it came from (without feedback), updating a
posterior over ponds after each observation. The free parameter `lambda1`
controls how strongly each observation updates belief — the probability
that the newly observed colour matches the previously-inferred pond's
dominant colour, versus being spread across the alternatives. Smaller
`lambda1` values imply slower belief updates, requiring more confirming
evidence before confidence increases.

Free parameters: `lambda1` (belief-update rate, in `[0, 1]`).

```python
import numpy as np
from pyem.models.bayes import bayes_sim, bayes_fit
sim = bayes_sim(np.array([[0.6]]), nblocks=3, ntrials=8, seed=0)
print(bayes_fit(np.array([0.0]), sim["choices"][0], sim["observations"][0], output="all")["nll"])
```

::: pyem.models.bayes.bayes_sim
::: pyem.models.bayes.bayes_fit
::: pyem.models.bayes.bayes_model

## Drift-diffusion models (pyem.models.ddm)

`pyem.models.ddm` implements drift-diffusion models (DDMs) for two
value-based choice tasks, each with a four-parameter and a
seven-parameter ("full DDM") form — four models in total, sharing a single
Wiener first-passage-time (WFPT) likelihood (Navarro & Fuss, 2009):

- **High-vs-low value** (`ddm4` / `ddm7`) — each trial offers two certain
  amounts and the agent should choose the higher-valued one. Drift is
  driven by the value gap and points toward the upper (correct/high)
  boundary: `v = v_coef * (value_high - value_low) >= 0`. Upper = chose
  high (correct), lower = chose low (error).
- **Safe-vs-risky gamble** (`ddm4_lotto` / `ddm7_lotto`) — a risky gamble
  (win probability `p`, payoff `payoff`, so `EV_risky = p * payoff`) is
  pitted against a safe certain amount. Drift is the risky-minus-safe value
  difference, `v = v_coef * (EV_risky - safe)`, and can point either way.
  Upper = risky (`choice = 1`), lower = safe (`choice = 0`).

Each task has two parameterizations. The **four-parameter** models
(`ddm4` / `ddm4_lotto`) use `[v_coef, a, t0, z]` (drift scaling, boundary
separation, non-decision time, relative start-point bias). The
**seven-parameter** models (`ddm7` / `ddm7_lotto`) add the three
across-trial variability parameters of the "full" diffusion model
(Ratcliff & Rouder, 1998; Ratcliff & Tuerlinckx, 2002): `sv` (SD of a
Normal on the trial drift), `st` (full width of a Uniform on the
non-decision time), and `sz` (full width of a Uniform on the relative
start point). Setting `sv = st = sz = 0` reduces each seven-parameter
model exactly to its four-parameter sibling. Drift variability `sv` is
marginalized analytically in closed form (Ratcliff & Tuerlinckx, 2002);
`st` and `sz` are integrated numerically by Gauss-Legendre quadrature.

Because the seven-parameter variability parameters are only weakly
identified for recovery in these value-based designs (see the module
docstring's Recoverability note for a worked example), **`ddm4` and
`ddm4_lotto` are recommended for parameter recovery**; the
seven-parameter models are retained for likelihood evaluation and for
simulating richer generative designs.

```python
import numpy as np
from pyem.models.ddm import ddm4_sim, ddm4_fit
sim = ddm4_sim(np.array([[2.0, 1.5, 0.2, 0.5]]), ntrials=60, seed=0)
print(ddm4_fit(np.array([2.0, 0.0, 0.0, 0.0]), sim["rt"][0], sim["choice"][0], sim["value_high"][0], sim["value_low"][0], output="all")["nll"])
```

### Shared transforms and likelihood

::: pyem.models.ddm.t0_xform
::: pyem.models.ddm.a_xform
::: pyem.models.ddm.sv_xform
::: pyem.models.ddm.st_xform
::: pyem.models.ddm.sz_xform
::: pyem.models.ddm.wfpt_sv_logpdf
::: pyem.models.ddm.wfpt_logpdf

### High-vs-low value task

::: pyem.models.ddm.ddm4_sim
::: pyem.models.ddm.ddm4_fit
::: pyem.models.ddm.ddm4_model
::: pyem.models.ddm.ddm7_sim
::: pyem.models.ddm.ddm7_fit
::: pyem.models.ddm.ddm7_sim_paths
::: pyem.models.ddm.ddm7_model

### Safe-vs-risky gamble task

::: pyem.models.ddm.ddm4_lotto_sim
::: pyem.models.ddm.ddm4_lotto_fit
::: pyem.models.ddm.ddm4_lotto_model
::: pyem.models.ddm.ddm7_lotto_sim
::: pyem.models.ddm.ddm7_lotto_fit
::: pyem.models.ddm.ddm7_lotto_sim_paths
::: pyem.models.ddm.ddm7_lotto_model
