# Prospect Theory & Discounting Models

This page covers two families of binary-choice models: `pyem.models.pt`
(prospect theory, certain amount vs. gamble) and `pyem.models.discounting`
(nine models spanning social, temporal, probability, effort, and prosocial
effort discounting). All models here follow the same overall pattern: a
subjective value is computed for each option, and choice probability is a
logistic (sigmoid/softmax) function of the value difference.

## Prospect theory (pyem.models.pt)

`pyem.models.pt` implements cumulative prospect theory (Tversky & Kahneman,
1992) for a certain-amount-vs-gamble choice task. Each trial presents a sure
`certain` amount against a two-outcome `gamble` whose outcomes occur with
probabilities `p1`/`p2 = 1 - p1` (outcomes may be mixed gain/loss).

Subjective value combines two components:

- **Value function** — a power function with separate curvature for gains
  and losses, plus a loss-aversion multiplier:
  `v(x) = x**alpha` for `x >= 0`, `v(x) = -lambda * (-x)**beta` for `x < 0`.
- **Probability weighting** — the one-parameter Tversky & Kahneman (1992)
  form: `w(p) = p**gamma / (p**gamma + (1-p)**gamma)**(1/gamma)`.

The gamble's value is `V_g = w(p1)*v(o1) + w(p2)*v(o2)`; the certain
option's value is `V_c = v(certain)`. Choices follow a logistic (softmax)
rule with inverse-temperature `mu`: `P(choose gamble) = expit(mu * (V_g -
V_c))`.

Free parameters: `alpha` (gain curvature), `beta` (loss curvature), `lambda`
(loss aversion), `gamma` (probability-weighting curvature), `mu` (choice
temperature).

```python
import numpy as np
from pyem.models.pt import pt_sim, pt_fit
nat = np.array([[0.8, 0.8, 1.5, 0.7, 2.0]])   # alpha,beta,lambda,gamma,mu
sim = pt_sim(nat, ntrials=60, seed=0)
print(pt_fit(np.zeros(5), sim["gamble"][0], sim["probs"][0], sim["certain"][0], sim["choice"][0], output="all")["nll"])
```

::: pyem.models.pt.pt_weight
::: pyem.models.pt.pt_value
::: pyem.models.pt.pt_sim
::: pyem.models.pt.pt_fit
::: pyem.models.pt.pt_model

## Discounting (pyem.models.discounting)

`pyem.models.discounting` implements nine binary-choice discounting models
across five task families, all sharing the same generative shape: a choice
between a "baseline" option (undiscounted) and a "discounted" option whose
value shrinks as a function of a block-level discounting variable. As that
variable grows, the discounted option's value falls and the indifference
point sweeps down a ladder of payouts, which is what identifies the discount
rate.

- **Social discounting** (`sd_*`) — the discounted option is a prosocial
  split whose value to the recipient falls as social distance `N` grows.
  Four variants: `sd_hyp_wk` (hyperbolic, with a free other-regarding weight
  `w_other`), `sd_hyp_k` (hyperbolic, `w_other` fixed at 1), `sd_par_k`
  (parabolic: `U_other(N) = r_other - k*N**2`), `sd_lin_k` (linear:
  `U_other(N) = r_other - k*N`).
- **Temporal (delay) discounting** (`td_hyp_k`) — smaller-sooner vs.
  larger-later: `V_later(D) = r_later / (1 + k*D)` (Mazur, 1987).
- **Probability discounting** (`prd_hyp_k`) — certain-small vs. risky-large:
  `V_risky(p) = r_risky / (1 + k*theta)`, discounted by the odds against
  winning `theta = (1-p)/p` (Rachlin, Raineri, & Cross, 1991).
- **Effort discounting** (`ed_par_k`) — low-effort-small vs. high-effort-large:
  `V_high(E) = r_high - k*E**2` (accelerating/parabolic effort cost).
- **Prosocial effort discounting** (`ped_par_k`, `ped_par_2k`) — the same
  effort-discounting task run in blocks where the reward benefits either the
  chooser ("self") or a social target ("other"); `ped_par_k` fits a single
  shared `k`, while `ped_par_2k` fits separate `k_self`/`k_other` rates
  (Lockwood et al., 2017).

Choice probability in every model is `p(discounted option) =
sigmoid(V_discounted - V_baseline)`.

```python
import numpy as np
from pyem.models.discounting import sd_hyp_k_sim, sd_hyp_k_fit
sim = sd_hyp_k_sim(np.array([[0.1]]), seed=0)
print(sd_hyp_k_fit(np.array([0.0]), sim["choices"][0], sim["payouts"][0], sim["social_dists"][0], output="all")["nll"])
```

```python
import numpy as np
from pyem.models.discounting import td_hyp_k_sim, td_hyp_k_fit
sim = td_hyp_k_sim(np.array([[0.05]]), seed=0)
print(td_hyp_k_fit(np.array([0.0]), sim["choices"][0], sim["payouts"][0], sim["delays"][0], output="all")["nll"])
```

```python
import numpy as np
from pyem.models.discounting import ped_par_2k_sim, ped_par_2k_fit
sim = ped_par_2k_sim(np.array([[0.5, 0.8]]), seed=0)
print(ped_par_2k_fit(np.array([0.0, 0.0]), sim["choices"][0], sim["payouts"][0], sim["effort_levels"][0], sim["beneficiary"][0], output="all")["nll"])
```

### Social discounting

::: pyem.models.discounting.sd_hyp_wk_sim
::: pyem.models.discounting.sd_hyp_wk_fit
::: pyem.models.discounting.sd_hyp_wk_model
::: pyem.models.discounting.sd_hyp_k_sim
::: pyem.models.discounting.sd_hyp_k_fit
::: pyem.models.discounting.sd_hyp_k_model
::: pyem.models.discounting.sd_par_k_sim
::: pyem.models.discounting.sd_par_k_fit
::: pyem.models.discounting.sd_par_k_model
::: pyem.models.discounting.sd_lin_k_sim
::: pyem.models.discounting.sd_lin_k_fit
::: pyem.models.discounting.sd_lin_k_model

### Temporal (delay) discounting

::: pyem.models.discounting.td_hyp_k_sim
::: pyem.models.discounting.td_hyp_k_fit
::: pyem.models.discounting.td_hyp_k_model

### Probability discounting

::: pyem.models.discounting.prd_hyp_k_sim
::: pyem.models.discounting.prd_hyp_k_fit
::: pyem.models.discounting.prd_hyp_k_model

### Effort discounting

::: pyem.models.discounting.ed_par_k_sim
::: pyem.models.discounting.ed_par_k_fit
::: pyem.models.discounting.ed_par_k_model

### Prosocial effort discounting

::: pyem.models.discounting.ped_par_k_sim
::: pyem.models.discounting.ped_par_k_fit
::: pyem.models.discounting.ped_par_k_model
::: pyem.models.discounting.ped_par_2k_sim
::: pyem.models.discounting.ped_par_2k_fit
::: pyem.models.discounting.ped_par_2k_model
