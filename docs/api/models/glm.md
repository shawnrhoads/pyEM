# GLM / Regression Models

`pyem.models.glm` implements linear and logistic regression models of
trial-level behavior, including decay (weighted history) and AR(1)
variants. As elsewhere in `pyem`, free parameters are fit in **Gaussian
(unbounded) space**, with decay/AR coefficients mapped to their natural
ranges inside `*_fit` (`norm2alpha` for `gamma`, identity for `phi`, bounds
checked directly).

!!! note "`*_sim` returns `(X, Y)`, not a dict"
    Unlike the RL model families, every `*_sim` function in this module
    returns a plain `(X, Y)` tuple rather than a dict. This means GLM models
    are **not** compatible with `EMModel.recover()` (which expects a dict of
    named simulation outputs) — fit GLMs directly against simulated `(X, Y)`
    data instead, as in the snippets below.

### glm — Gaussian linear regression

Standard Gaussian linear regression: `Y` is generated as a linear
combination of predictors `X` plus Gaussian noise. Free parameters:
regression weights (intercept + covariates).

```python
import numpy as np
from pyem.models.glm import glm_sim, glm_fit

X, Y = glm_sim(np.zeros((5, 3)), ntrials=50, seed=0)
print(glm_fit(np.zeros(3), X[0], Y[0], output="all")["nll"])
```

::: pyem.models.glm.glm_sim
::: pyem.models.glm.glm_fit
::: pyem.models.glm.glm_model

### glm_decay — regression with exponentially discounted predictors

Gaussian linear regression where the current prediction is a weighted sum
of the current and previous trials' predictors, discounted by `gamma` per
step back. Free parameters: regression weights, `gamma` (discount factor,
in `[0, 1]`, trailing parameter).

```python
import numpy as np
from pyem.models.glm import glm_decay_sim, glm_decay_fit

X, Y = glm_decay_sim(np.zeros((5, 4)), ntrials=50, seed=0)
print(glm_decay_fit(np.zeros(4), X[0], Y[0], output="all")["nll"])
```

::: pyem.models.glm.glm_decay_sim
::: pyem.models.glm.glm_decay_fit
::: pyem.models.glm.glm_decay_model

### logit — logistic regression

Standard logistic regression: `Y` (0/1) is drawn from a Bernoulli
distribution with probability given by the logistic (expit) link applied to
a linear combination of predictors `X`. Free parameters: regression weights
(intercept + covariates).

```python
import numpy as np
from pyem.models.glm import logit_sim, logit_fit

X, Y = logit_sim(np.zeros((5, 3)), ntrials=50, seed=0)
print(logit_fit(np.zeros(3), X[0], Y[0], output="all")["nll"])
```

::: pyem.models.glm.logit_sim
::: pyem.models.glm.logit_fit
::: pyem.models.glm.logit_model

### logit_decay — logistic regression with exponentially discounted predictors

Mirrors `glm_decay`'s discounting scheme applied to a logistic link function
instead of an identity link. Free parameters: regression weights, `gamma`
(discount factor, in `[0, 1]`, trailing parameter).

```python
import numpy as np
from pyem.models.glm import logit_decay_sim, logit_decay_fit

X, Y = logit_decay_sim(np.zeros((5, 4)), ntrials=50, seed=0)
print(logit_decay_fit(np.zeros(4), X[0], Y[0], output="all")["nll"])
```

::: pyem.models.glm.logit_decay_sim
::: pyem.models.glm.logit_decay_fit
::: pyem.models.glm.logit_decay_model

### glm_ar — regression with an AR(1) term

Gaussian linear regression with an AR(1) autoregressive term on the
outcome: `y_t = lin_t + phi * y_(t-1) + noise`. Free parameters: regression
weights, `phi` (AR(1) coefficient, in `(-1, 1)`, trailing parameter).

```python
import numpy as np
from pyem.models.glm import glm_ar_sim, glm_ar_fit

X, Y = glm_ar_sim(np.zeros((5, 4)), ntrials=50, seed=0)
print(glm_ar_fit(np.zeros(4), X[0], Y[0], output="all")["nll"])
```

::: pyem.models.glm.glm_ar_sim
::: pyem.models.glm.glm_ar_fit
::: pyem.models.glm.glm_ar_model
