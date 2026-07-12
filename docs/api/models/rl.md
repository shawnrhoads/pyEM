# Reinforcement Learning Models

`pyem.models.rl_mf` and `pyem.models.rl_mb` implement two families of
reinforcement-learning models. As with every model in `pyem`, free parameters
are fit in **Gaussian (unbounded) space** and mapped to their natural ranges
inside each `*_fit` function (via `norm2beta`/`norm2alpha` or, for
`rl_mb`, a plain `exp`/logistic transform — see below); `*_sim` functions take
**natural**-space parameters directly.

The `rl_mf` family implements 2-option Rescorla–Wagner variants: `rw1a1b` and
`rw2a1b` are the standard single- and split-learning-rate models, while
`rw3a1b` and `rw4a1b` add extra outcome channels for social/vicarious learning
(`rw3a1b`: Lockwood et al., 2016; `rw4a1b`: Rhoads et al., 2025). The `rl_mb`
family implements the Daw et al. (2011) two-step task: a pure model-free
SARSA(&lambda;) learner, a pure model-based Bellman learner, and the hybrid
that mixes the two with weight &omega;.

## Model-free (pyem.models.rl_mf)

### rw1a1b — single learning rate

A 2-option Rescorla–Wagner model with a single learning rate and softmax
choice. Free parameters: `beta` (inverse temperature), `alpha` (learning
rate).

```python
import numpy as np
from pyem.models.rl_mf import rw1a1b_sim, rw1a1b_fit

sim = rw1a1b_sim(np.array([[3.0, 0.4]]), nblocks=3, ntrials=24, seed=0)
print(rw1a1b_fit(np.zeros(2), sim["choices"][0], sim["rewards"][0], output="all")["nll"])
```

::: pyem.models.rl_mf.rw1a1b_sim
::: pyem.models.rl_mf.rw1a1b_fit
::: pyem.models.rl_mf.rw1a1b_model

### rw2a1b — separate gain/loss learning rates

A 2-option Rescorla–Wagner model with separate learning rates for positive vs
negative prediction errors (valence bias). Free parameters: `beta`,
`alpha_pos`, `alpha_neg`.

```python
import numpy as np
from pyem.models.rl_mf import rw2a1b_sim, rw2a1b_fit

sim = rw2a1b_sim(np.array([[3.0, 0.5, 0.3]]), nblocks=3, ntrials=24, seed=0)
print(rw2a1b_fit(np.zeros(3), sim["choices"][0], sim["rewards"][0], output="all")["nll"])
```

::: pyem.models.rl_mf.rw2a1b_sim
::: pyem.models.rl_mf.rw2a1b_fit
::: pyem.models.rl_mf.rw2a1b_model

### rw3a1b — self/other/no-one outcome channels

A 2-option task with three binary outcome channels (self, other, no-one);
Rescorla–Wagner learning combines the three prediction errors into a single
expected-value update (Lockwood et al., 2016). Free parameters: `beta`,
`alpha_self`, `alpha_other`, `alpha_noone`. `*_fit` takes `(choices,
rewards_dict)`, where `rewards_dict` has `rewards_self`/`rewards_other`/
`rewards_noone` keys — the `*_sim` output's `"rewards"` entry is already a
per-subject list of such dicts, ready for `sim["rewards"][s]`.

```python
import numpy as np
from pyem.models.rl_mf import rw3a1b_sim, rw3a1b_fit

sim = rw3a1b_sim(np.array([[3.0, 0.5, 0.5, 0.5]]), nblocks=9, ntrials=16, seed=0)
print(rw3a1b_fit(np.zeros(4), sim["choices"][0], sim["rewards"][0], output="all")["nll"])
```

::: pyem.models.rl_mf.rw3a1b_sim
::: pyem.models.rl_mf.rw3a1b_fit
::: pyem.models.rl_mf.rw3a1b_model

### rw4a1b — self/other × positive/negative outcome channels

A 4-option task where each trial shows a pair of options; one shared inverse
temperature and four learning rates split by outcome recipient (self/other)
and valence (positive/negative) (Rhoads et al., 2025). Free parameters:
`beta`, `alpha_self_pos`, `alpha_self_neg`, `alpha_other_pos`,
`alpha_other_neg`. `*_fit` takes `(choices, outcomes_self, outcomes_other,
option_pairs)`.

```python
import numpy as np
from pyem.models.rl_mf import rw4a1b_sim, rw4a1b_fit

sim = rw4a1b_sim(np.array([[3.0, 0.5, 0.5, 0.5, 0.5]]), nblocks=6, ntrials=20, seed=0)
print(rw4a1b_fit(np.zeros(5), sim["choices"][0], sim["outcomes_self"][0], sim["outcomes_other"][0], sim["option_pairs"][0], output="all")["nll"])
```

::: pyem.models.rl_mf.rw4a1b_sim
::: pyem.models.rl_mf.rw4a1b_fit
::: pyem.models.rl_mf.rw4a1b_model
::: pyem.models.rl_mf.gen_rnd_blocks

## Model-based / two-step (pyem.models.rl_mb)

These three learners share the same trial-by-trial two-step update/choice
logic and only differ in which terms are free vs. fixed. Following the
authors' original MATLAB code (`llm2b2alr.m`), the Gaussian→natural
transforms deviate from the rest of `pyem` in two ways: `beta1`/`beta2` use a
plain `exp(x)` (no finite ceiling, unlike `norm2beta`), and the first-stage
stickiness `r` is passed through unchanged (identity transform).

### sarsa_lambda — pure model-free SARSA(&lambda;)

The `omega = 0` special case of the hybrid: first-stage values are updated
via a &lambda;-weighted, stage-skipping SARSA(&lambda;) rule rather than the
model-based Bellman backup. Free parameters (natural-space order):
`beta1, beta2, alpha1, alpha2, lambda, r` (6 params).

```python
import numpy as np
from pyem.models.rl_mb import sarsa_lambda_sim, sarsa_lambda_fit

nat = np.array([[3., 3., 0.5, 0.5, 0.5, 0.0]])
sim = sarsa_lambda_sim(nat, ntrials=80, seed=0)
print(sarsa_lambda_fit(np.zeros(6), sim["choices1"][0], sim["states2"][0], sim["choices2"][0], sim["rewards"][0], output="all")["nll"])
```

::: pyem.models.rl_mb.sarsa_lambda_sim
::: pyem.models.rl_mb.sarsa_lambda_fit
::: pyem.models.rl_mb.sarsa_lambda_model

### model_based — pure model-based Bellman learner

The `omega = 1` special case of the hybrid: first-stage values are
recomputed each trial from the (count-estimated) transition structure and
learned second-stage values via the Bellman equation. Because the
model-free first-stage values never enter the choice, `alpha1` and `lambda`
drop out. Free parameters: `beta1, beta2, alpha2, r` (4 params).

```python
import numpy as np
from pyem.models.rl_mb import model_based_sim, model_based_fit

nat = np.array([[3., 3., 0.5, 0.0]])
sim = model_based_sim(nat, ntrials=80, seed=0)
print(model_based_fit(np.zeros(4), sim["choices1"][0], sim["states2"][0], sim["choices2"][0], sim["rewards"][0], output="all")["nll"])
```

::: pyem.models.rl_mb.model_based_sim
::: pyem.models.rl_mb.model_based_fit
::: pyem.models.rl_mb.model_based_model

### hybrid_mbmf — mixed model-based / model-free

First-stage net values are a weighted sum of model-based (Bellman) and
model-free (SARSA(&lambda;)) values, `w*Q_MB + (1-w)*Q_MF`, plus first-stage
perseveration; nests the pure model-free (`omega=0`) and model-based
(`omega=1`) learners above. Free parameters: `beta1, beta2, alpha1, alpha2,
lambda, omega, r` (7 params).

```python
import numpy as np
from pyem.models.rl_mb import hybrid_mbmf_sim, hybrid_mbmf_fit

nat = np.array([[3.,3.,0.5,0.5,0.5,0.5,0.0]])
sim = hybrid_mbmf_sim(nat, ntrials=80, seed=0)
print(hybrid_mbmf_fit(np.zeros(7), sim["choices1"][0], sim["states2"][0], sim["choices2"][0], sim["rewards"][0], output="all")["nll"])
```

::: pyem.models.rl_mb.hybrid_mbmf_sim
::: pyem.models.rl_mb.hybrid_mbmf_fit
::: pyem.models.rl_mb.hybrid_mbmf_model
