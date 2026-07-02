"""pyem model: social discounting over a binary self/other allocation task.

On each trial, an agent chooses between:
  - a "selfish" option: keep everything, give the social target nothing.
  - a "prosocial" option: split the pot, with the target's share landing at
    social distance N (1 = closest, 100 = a stranger).

Four independent ModelSpec variants are defined below, each with its own
fully-written-out sim/fit pair (no shared "core" dispatcher) so the trial
loop is visible in each function rather than hidden behind a generic engine:

    sd_hyp_wk : hyperbolic discounting, 2 free params (w_other, k)
    sd_hyp_k  : hyperbolic discounting, 1 free param  (k;  w_other fixed at 1)
    sd_par_k  : parabolic discounting,  1 free param  (k;  w_other fixed at 1)
    sd_lin_k  : linear discounting,     1 free param  (k;  w_other fixed at 1)

Choice rule (all variants): the probability of choosing the prosocial option
is a logistic (sigmoid) function of the value difference, the standard
choice function in the discounting literature:

    p(prosocial) = sigmoid(V_prosocial - V_selfish) = 1 / (1 + exp(-delta_V))

with per-option value V = r_self + U_other(N), where r_self is the chooser's
own payout under that option. There is no free inverse temperature: delta_V
is in dollar units (scaled by w_other where that weight is free).

Axis convention for payouts:
    payouts[trial, choice, target]
where choice 0='selfish', 1='prosocial' and target 0='self', 1='other'.

Discount shapes (R = w_other * r_other, the target's weighted, undiscounted
share; r_other is 0 for the selfish option):
    hyperbolic:  U_other(N) = R / (1 + k*N)
    linear:      U_other(N) = R - k*N        (only where R > 0, else 0)
    parabolic:   U_other(N) = R - k*N**2     (only where R > 0, else 0)

The "only where R > 0" clause matters: the selfish option always has
r_other = 0, so R = 0 for it. Hyperbolic discounting handles this for free
(0 / anything = 0). The subtractive forms do not -- applying `0 - k*N` to the
selfish option's target-share would inject a `-k*N` penalty that also
appears (and exactly cancels) in `delta_V`, making N have zero effect on
choice. Gating the subtraction on R > 0 avoids that degeneracy and keeps
"nothing offered" equal to "nothing discounted" for every discount shape.
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from pyem.core.modelspec import ModelSpec
from pyem.utils.math import norm2beta, calc_fval

# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
DEFAULT_SOCIAL_DISTS = np.array([1, 2, 5, 10, 20, 50, 100], dtype=float)

TASK1_PAYOUTS = np.array([
    [[155.0,   0.0], [ 75.0,  75.0]],
    [[145.0,   0.0], [ 75.0,  75.0]],
    [[135.0,   0.0], [ 75.0,  75.0]],
    [[125.0,   0.0], [ 75.0,  75.0]],
    [[115.0,   0.0], [ 75.0,  75.0]],
    [[105.0,   0.0], [ 75.0,  75.0]],
    [[ 95.0,   0.0], [ 75.0,  75.0]],
    [[ 85.0,   0.0], [ 75.0,  75.0]],
    [[ 75.0,   0.0], [ 75.0,  75.0]],
], dtype=float)

TASK2_PAYOUTS = np.array([
    [[ 95.0,   0.0], [  0.0, 105.0]],
    [[ 85.0,   0.0], [  0.0, 105.0]],
    [[ 75.0,   0.0], [  0.0, 105.0]],
    [[ 65.0,   0.0], [  0.0, 105.0]],
    [[ 55.0,   0.0], [  0.0, 105.0]],
    [[ 45.0,   0.0], [  0.0, 105.0]],
    [[ 35.0,   0.0], [  0.0, 105.0]],
    [[ 25.0,   0.0], [  0.0, 105.0]],
    [[ 15.0,   0.0], [  0.0, 105.0]],
], dtype=float)

CHOICE_LABELS = np.array(["selfish", "prosocial"], dtype=object)
CHOICE_MAP = {
    "selfish": 0, "prosocial": 1,
    "self": 0, "other": 1,
    "s": 0, "p": 1,
    0: 0, 1: 1,
}


# -----------------------------------------------------------------------------
# Shared input-parsing helpers (boilerplate shape/validity checks, not part of
# the discounting computation itself -- every model below calls these).
# -----------------------------------------------------------------------------
def _prepare_social_inputs(
    payouts: np.ndarray | None = None,
    social_dists: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if social_dists is None:
        social_dists = DEFAULT_SOCIAL_DISTS.copy()
    social_dists = np.asarray(social_dists, dtype=float)

    if social_dists.ndim != 1:
        raise ValueError("social_dists must be a 1D array of social distances")
    if social_dists.size == 0:
        raise ValueError("social_dists must contain at least one block distance")
    if np.any(social_dists <= 0.0):
        raise ValueError("social_dists values must be > 0")

    if payouts is None:
        payouts = TASK1_PAYOUTS.copy()
    payouts = np.asarray(payouts, dtype=float)

    if payouts.ndim == 3:
        if payouts.shape[1:] != (2, 2):
            raise ValueError("3D payouts must have shape (ntrials, 2, 2)")
        nblocks = social_dists.size
        ntrials = payouts.shape[0]
        payouts_bt = np.broadcast_to(payouts[None, ...], (nblocks, ntrials, 2, 2)).copy()
    elif payouts.ndim == 4:
        if payouts.shape[2:] != (2, 2):
            raise ValueError("4D payouts must have shape (nblocks, ntrials, 2, 2)")
        if payouts.shape[0] == 1:
            payouts_bt = np.broadcast_to(payouts, (social_dists.size, payouts.shape[1], 2, 2)).copy()
        elif payouts.shape[0] == social_dists.size:
            payouts_bt = payouts.copy()
        else:
            raise ValueError("If payouts is 4D, its first dimension must equal len(social_dists) or be 1")
    else:
        raise ValueError("payouts must have shape (ntrials, 2, 2) or (nblocks, ntrials, 2, 2)")

    return payouts_bt, social_dists


def _choices_to_idx(choices: np.ndarray, nblocks: int, ntrials: int) -> np.ndarray:
    arr = np.asarray(choices)
    if arr.shape != (nblocks, ntrials):
        raise ValueError("choices must have shape (nblocks, ntrials)")

    if np.issubdtype(arr.dtype, np.number):
        idx = arr.astype(int, copy=False)
        if not np.isin(idx, [0, 1]).all():
            raise ValueError("Numeric choices must be coded as 0=selfish or 1=prosocial")
        return idx

    idx = np.empty((nblocks, ntrials), dtype=int)
    for b in range(nblocks):
        for t in range(ntrials):
            key = arr[b, t]
            if isinstance(key, str):
                key = key.strip().lower()
            if key not in CHOICE_MAP:
                raise ValueError("choices must be strings {'selfish','prosocial'} or numeric {0,1}")
            idx[b, t] = CHOICE_MAP[key]
    return idx


def _expand_for_subjects(payouts_bt: np.ndarray, social_dists: np.ndarray, nsubjects: int) -> tuple[np.ndarray, np.ndarray]:
    nblocks, ntrials = payouts_bt.shape[:2]
    payouts_subj = np.broadcast_to(payouts_bt[None, ...], (nsubjects, nblocks, ntrials, 2, 2)).copy()
    social_dists_subj = np.broadcast_to(social_dists[None, :], (nsubjects, nblocks)).copy()
    return payouts_subj, social_dists_subj


# -----------------------------------------------------------------------------
# Output-dict builders. These just assemble already-computed arrays into the
# dict shape every sim/fit below returns -- not part of the discounting
# computation, so they're factored out even though the trial loops are not.
# -----------------------------------------------------------------------------
def _sim_output(params, payouts_subj, social_dists_subj, U_self, U_other, EV, delta_V,
                 p_prosocial, choices, ch_prob, choices_prosocial, outcomes, nll) -> dict:
    return {
        "params": params.copy(),
        "payouts": payouts_subj,
        "social_dists": social_dists_subj,
        "U_self": U_self,
        "U_other": U_other,
        "EV": EV,
        "delta_V": delta_V,
        "p_prosocial": p_prosocial,
        "choices": choices,
        "ch_prob": ch_prob,
        "choices_prosocial": choices_prosocial,
        "outcomes": outcomes,
        "nll": nll,
        "nll_total": nll.sum(axis=(1, 2)),
    }


def _fit_output(params_natural, payouts_bt, social_dists, U_self, U_other, EV, delta_V,
                 p_prosocial, choices, ch_prob, choices_prosocial, outcomes, nll, nll_trialwise) -> dict:
    return {
        "params": params_natural,
        "payouts": payouts_bt,
        "social_dists": social_dists.copy(),
        "U_self": U_self,
        "U_other": U_other,
        "EV": EV,
        "delta_V": delta_V,
        "p_prosocial": p_prosocial,
        "choices": np.asarray(choices, dtype=object).copy(),
        "ch_prob": ch_prob,
        "choices_prosocial": choices_prosocial,
        "outcomes": outcomes,
        "nll": nll,
        "nll_trialwise": nll_trialwise,
    }


# =============================================================================
# sd_hyp_wk -- hyperbolic discounting, 2 free parameters (w_other, k)
#   U_other(N) = w_other * r_other / (1 + k*N)
# =============================================================================
def sd_hyp_wk_sim(params: np.ndarray, payouts=TASK1_PAYOUTS, social_dists=DEFAULT_SOCIAL_DISTS) -> dict:
    """Simulate the 2-parameter hyperbolic social discounting model (w_other, k free)."""
    payouts_bt, social_dists = _prepare_social_inputs(payouts=payouts, social_dists=social_dists)
    params = np.asarray(params, dtype=float)
    if params.ndim != 2 or params.shape[1] != 2:
        raise ValueError("params must have shape (nsubjects, 2): columns [w_other, k]")

    nsubjects = params.shape[0]
    nblocks, ntrials = payouts_bt.shape[:2]
    payouts_subj, social_dists_subj = _expand_for_subjects(payouts_bt, social_dists, nsubjects)
    rng = np.random.default_rng()

    choices = np.empty((nsubjects, nblocks, ntrials), dtype=object)
    choices_prosocial = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    ch_prob = np.zeros((nsubjects, nblocks, ntrials, 2), dtype=float)
    p_prosocial = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    outcomes = np.zeros((nsubjects, nblocks, ntrials, 2), dtype=float)
    U_self = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    U_other = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    EV = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    delta_V = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    nll = np.zeros((nsubjects, nblocks, ntrials), dtype=float)

    all_w_other = params[:, 0]
    all_k = params[:, 1]
    if not (all_w_other > 0.0).all():
        raise ValueError("w_other must be > 0")
    if not (all_k > 0.0).all():
        raise ValueError("k must be > 0")

    for s in range(nsubjects):
        w_other = float(all_w_other[s])
        k = float(all_k[s])

        for b in range(nblocks):
            N = float(social_dists[b])

            for t in range(ntrials):
                trial_payouts = payouts_bt[b, t, :, :]
                r_self = trial_payouts[:, 0]
                r_other = trial_payouts[:, 1]

                U_self_opts = r_self.copy()
                U_other_opts = (w_other * r_other) / (1.0 + k * N)
                V_opts = U_self_opts + U_other_opts

                dv = V_opts[1] - V_opts[0]
                p1 = float(expit(dv))
                p = np.array([1.0 - p1, p1])
                c = int(rng.choice([0, 1], p=p))

                ch_prob[s, b, t, :] = p
                p_prosocial[s, b, t] = p1
                delta_V[s, b, t] = dv
                choices[s, b, t] = CHOICE_LABELS[c]
                choices_prosocial[s, b, t] = float(c == 1)
                outcomes[s, b, t, :] = trial_payouts[c, :]
                U_self[s, b, t] = U_self_opts[c]
                U_other[s, b, t] = U_other_opts[c]
                EV[s, b, t] = V_opts[c]
                nll[s, b, t] = -np.log(p[c] + 1e-12)

    return _sim_output(params, payouts_subj, social_dists_subj, U_self, U_other, EV, delta_V,
                        p_prosocial, choices, ch_prob, choices_prosocial, outcomes, nll)


def sd_hyp_wk_fit(params, choices, payouts, social_dists, prior=None, output: str = "npl"):
    """Fit the 2-parameter hyperbolic social discounting model. params: (2,) Gaussian-space [w_other, k]."""
    payouts_bt, social_dists = _prepare_social_inputs(payouts=payouts, social_dists=social_dists)
    nblocks, ntrials = payouts_bt.shape[:2]
    choice_idx = _choices_to_idx(choices, nblocks=nblocks, ntrials=ntrials)

    params = np.asarray(params, dtype=float)
    if params.shape != (2,):
        raise ValueError("params must have shape (2,) in normalized space: [w_other, k]")

    w_other = float(norm2beta(params[0]))
    k = float(norm2beta(params[1]))
    if not (w_other > 0.0):
        return 1e7
    if not (k > 0.0):
        return 1e7

    U_self = np.zeros((nblocks, ntrials), dtype=float)
    U_other = np.zeros((nblocks, ntrials), dtype=float)
    EV = np.zeros((nblocks, ntrials), dtype=float)
    delta_V = np.zeros((nblocks, ntrials), dtype=float)
    p_prosocial = np.zeros((nblocks, ntrials), dtype=float)
    ch_prob = np.zeros((nblocks, ntrials, 2), dtype=float)
    outcomes = np.zeros((nblocks, ntrials, 2), dtype=float)
    choices_prosocial = np.zeros((nblocks, ntrials), dtype=float)
    nll_trialwise = np.zeros((nblocks, ntrials), dtype=float)
    nll = 0.0

    for b in range(nblocks):
        N = float(social_dists[b])

        for t in range(ntrials):
            trial_payouts = payouts_bt[b, t, :, :]
            r_self = trial_payouts[:, 0]
            r_other = trial_payouts[:, 1]

            U_self_opts = r_self.copy()
            U_other_opts = (w_other * r_other) / (1.0 + k * N)
            V_opts = U_self_opts + U_other_opts

            dv = V_opts[1] - V_opts[0]
            p1 = float(expit(dv))
            p = np.array([1.0 - p1, p1])
            c = int(choice_idx[b, t])

            delta_V[b, t] = dv
            p_prosocial[b, t] = p1
            ch_prob[b, t, :] = p
            choices_prosocial[b, t] = float(c == 1)
            outcomes[b, t, :] = trial_payouts[c, :]
            U_self[b, t] = U_self_opts[c]
            U_other[b, t] = U_other_opts[c]
            EV[b, t] = V_opts[c]
            nll_trialwise[b, t] = -np.log(p[c] + 1e-12)
            nll += nll_trialwise[b, t]

    if output == "all":
        return _fit_output([w_other, k], payouts_bt, social_dists, U_self, U_other, EV, delta_V,
                            p_prosocial, choices, ch_prob, choices_prosocial, outcomes, nll, nll_trialwise)

    return calc_fval(nll, params, prior=prior, output=output)


sd_hyp_wk_desc = """Hyperbolic social discounting: U_other(N) = w_other*r_other / (1 + k*N).
The other-regarding weight is applied to the target's payoff, then discounted
hyperbolically as a function of social distance N.
p(prosocial) = sigmoid(V_prosocial - V_selfish).
Free parameters: w_other (other-regarding weight, >0), k (hyperbolic discount rate, >0)."""
sd_hyp_wk_id = "sd_hyp_wk"
sd_hyp_wk_spec = {"social_discounting": {"weight": ["w_other"], "discount": ["k"]}, "shape": "hyperbolic", "choice_rule": "sigmoid(delta_V)"}
sd_hyp_wk_model = ModelSpec(
    id=sd_hyp_wk_id, spec=sd_hyp_wk_spec, desc=sd_hyp_wk_desc.strip(),
    params=None, sim=sd_hyp_wk_sim, fit=sd_hyp_wk_fit,
)


# =============================================================================
# sd_hyp_k -- hyperbolic discounting, 1 free parameter (k; w_other fixed at 1)
#   U_other(N) = r_other / (1 + k*N)
# =============================================================================
def sd_hyp_k_sim(params: np.ndarray, payouts=TASK1_PAYOUTS, social_dists=DEFAULT_SOCIAL_DISTS) -> dict:
    """Simulate the 1-parameter hyperbolic social discounting model (k free)."""
    payouts_bt, social_dists = _prepare_social_inputs(payouts=payouts, social_dists=social_dists)
    params = np.asarray(params, dtype=float)
    if params.ndim != 2 or params.shape[1] != 1:
        raise ValueError("params must have shape (nsubjects, 1): column [k]")

    nsubjects = params.shape[0]
    nblocks, ntrials = payouts_bt.shape[:2]
    payouts_subj, social_dists_subj = _expand_for_subjects(payouts_bt, social_dists, nsubjects)
    rng = np.random.default_rng()

    choices = np.empty((nsubjects, nblocks, ntrials), dtype=object)
    choices_prosocial = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    ch_prob = np.zeros((nsubjects, nblocks, ntrials, 2), dtype=float)
    p_prosocial = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    outcomes = np.zeros((nsubjects, nblocks, ntrials, 2), dtype=float)
    U_self = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    U_other = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    EV = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    delta_V = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    nll = np.zeros((nsubjects, nblocks, ntrials), dtype=float)

    all_k = params[:, 0]
    if not (all_k > 0.0).all():
        raise ValueError("k must be > 0")

    for s in range(nsubjects):
        k = float(all_k[s])

        for b in range(nblocks):
            N = float(social_dists[b])

            for t in range(ntrials):
                trial_payouts = payouts_bt[b, t, :, :]
                r_self = trial_payouts[:, 0]
                r_other = trial_payouts[:, 1]

                U_self_opts = r_self.copy()
                U_other_opts = r_other / (1.0 + k * N)
                V_opts = U_self_opts + U_other_opts

                dv = V_opts[1] - V_opts[0]
                p1 = float(expit(dv))
                p = np.array([1.0 - p1, p1])
                c = int(rng.choice([0, 1], p=p))

                ch_prob[s, b, t, :] = p
                p_prosocial[s, b, t] = p1
                delta_V[s, b, t] = dv
                choices[s, b, t] = CHOICE_LABELS[c]
                choices_prosocial[s, b, t] = float(c == 1)
                outcomes[s, b, t, :] = trial_payouts[c, :]
                U_self[s, b, t] = U_self_opts[c]
                U_other[s, b, t] = U_other_opts[c]
                EV[s, b, t] = V_opts[c]
                nll[s, b, t] = -np.log(p[c] + 1e-12)

    return _sim_output(params, payouts_subj, social_dists_subj, U_self, U_other, EV, delta_V,
                        p_prosocial, choices, ch_prob, choices_prosocial, outcomes, nll)


def sd_hyp_k_fit(params, choices, payouts, social_dists, prior=None, output: str = "npl"):
    """Fit the 1-parameter hyperbolic social discounting model. params: (1,) Gaussian-space [k]."""
    payouts_bt, social_dists = _prepare_social_inputs(payouts=payouts, social_dists=social_dists)
    nblocks, ntrials = payouts_bt.shape[:2]
    choice_idx = _choices_to_idx(choices, nblocks=nblocks, ntrials=ntrials)

    params = np.asarray(params, dtype=float)
    if params.shape != (1,):
        raise ValueError("params must have shape (1,) in normalized space: [k]")

    k = float(norm2beta(params[0]))
    if not (k > 0.0):
        return 1e7

    U_self = np.zeros((nblocks, ntrials), dtype=float)
    U_other = np.zeros((nblocks, ntrials), dtype=float)
    EV = np.zeros((nblocks, ntrials), dtype=float)
    delta_V = np.zeros((nblocks, ntrials), dtype=float)
    p_prosocial = np.zeros((nblocks, ntrials), dtype=float)
    ch_prob = np.zeros((nblocks, ntrials, 2), dtype=float)
    outcomes = np.zeros((nblocks, ntrials, 2), dtype=float)
    choices_prosocial = np.zeros((nblocks, ntrials), dtype=float)
    nll_trialwise = np.zeros((nblocks, ntrials), dtype=float)
    nll = 0.0

    for b in range(nblocks):
        N = float(social_dists[b])

        for t in range(ntrials):
            trial_payouts = payouts_bt[b, t, :, :]
            r_self = trial_payouts[:, 0]
            r_other = trial_payouts[:, 1]

            U_self_opts = r_self.copy()
            U_other_opts = r_other / (1.0 + k * N)
            V_opts = U_self_opts + U_other_opts

            dv = V_opts[1] - V_opts[0]
            p1 = float(expit(dv))
            p = np.array([1.0 - p1, p1])
            c = int(choice_idx[b, t])

            delta_V[b, t] = dv
            p_prosocial[b, t] = p1
            ch_prob[b, t, :] = p
            choices_prosocial[b, t] = float(c == 1)
            outcomes[b, t, :] = trial_payouts[c, :]
            U_self[b, t] = U_self_opts[c]
            U_other[b, t] = U_other_opts[c]
            EV[b, t] = V_opts[c]
            nll_trialwise[b, t] = -np.log(p[c] + 1e-12)
            nll += nll_trialwise[b, t]

    if output == "all":
        return _fit_output([k], payouts_bt, social_dists, U_self, U_other, EV, delta_V,
                            p_prosocial, choices, ch_prob, choices_prosocial, outcomes, nll, nll_trialwise)

    return calc_fval(nll, params, prior=prior, output=output)


sd_hyp_k_desc = """Hyperbolic social discounting: U_other(N) = r_other / (1 + k*N).
The target's payoff is discounted hyperbolically as a function of social
distance N (w_other fixed at 1). p(prosocial) = sigmoid(V_prosocial - V_selfish).
Free parameter: k (hyperbolic discount rate, >0)."""
sd_hyp_k_id = "sd_hyp_k"
sd_hyp_k_spec = {"social_discounting": {"weight": [], "discount": ["k"]}, "shape": "hyperbolic", "choice_rule": "sigmoid(delta_V)"}
sd_hyp_k_model = ModelSpec(
    id=sd_hyp_k_id, spec=sd_hyp_k_spec, desc=sd_hyp_k_desc.strip(),
    params=None, sim=sd_hyp_k_sim, fit=sd_hyp_k_fit,
)


# =============================================================================
# sd_par_k -- parabolic discounting, 1 free parameter (k; w_other fixed at 1)
#   U_other(N) = r_other - k*N**2   (only where r_other > 0)
# =============================================================================
def sd_par_k_sim(params: np.ndarray, payouts=TASK1_PAYOUTS, social_dists=DEFAULT_SOCIAL_DISTS) -> dict:
    """Simulate the 1-parameter parabolic social discounting model (k free)."""
    payouts_bt, social_dists = _prepare_social_inputs(payouts=payouts, social_dists=social_dists)
    params = np.asarray(params, dtype=float)
    if params.ndim != 2 or params.shape[1] != 1:
        raise ValueError("params must have shape (nsubjects, 1): column [k]")

    nsubjects = params.shape[0]
    nblocks, ntrials = payouts_bt.shape[:2]
    payouts_subj, social_dists_subj = _expand_for_subjects(payouts_bt, social_dists, nsubjects)
    rng = np.random.default_rng()

    choices = np.empty((nsubjects, nblocks, ntrials), dtype=object)
    choices_prosocial = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    ch_prob = np.zeros((nsubjects, nblocks, ntrials, 2), dtype=float)
    p_prosocial = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    outcomes = np.zeros((nsubjects, nblocks, ntrials, 2), dtype=float)
    U_self = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    U_other = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    EV = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    delta_V = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    nll = np.zeros((nsubjects, nblocks, ntrials), dtype=float)

    all_k = params[:, 0]
    if not (all_k > 0.0).all():
        raise ValueError("k must be > 0")

    for s in range(nsubjects):
        k = float(all_k[s])

        for b in range(nblocks):
            N = float(social_dists[b])

            for t in range(ntrials):
                trial_payouts = payouts_bt[b, t, :, :]
                r_self = trial_payouts[:, 0]
                r_other = trial_payouts[:, 1]

                U_self_opts = r_self.copy()
                U_other_opts = np.where(r_other > 0.0, r_other - k * (N ** 2), 0.0)
                V_opts = U_self_opts + U_other_opts

                dv = V_opts[1] - V_opts[0]
                p1 = float(expit(dv))
                p = np.array([1.0 - p1, p1])
                c = int(rng.choice([0, 1], p=p))

                ch_prob[s, b, t, :] = p
                p_prosocial[s, b, t] = p1
                delta_V[s, b, t] = dv
                choices[s, b, t] = CHOICE_LABELS[c]
                choices_prosocial[s, b, t] = float(c == 1)
                outcomes[s, b, t, :] = trial_payouts[c, :]
                U_self[s, b, t] = U_self_opts[c]
                U_other[s, b, t] = U_other_opts[c]
                EV[s, b, t] = V_opts[c]
                nll[s, b, t] = -np.log(p[c] + 1e-12)

    return _sim_output(params, payouts_subj, social_dists_subj, U_self, U_other, EV, delta_V,
                        p_prosocial, choices, ch_prob, choices_prosocial, outcomes, nll)


def sd_par_k_fit(params, choices, payouts, social_dists, prior=None, output: str = "npl"):
    """Fit the 1-parameter parabolic social discounting model. params: (1,) Gaussian-space [k]."""
    payouts_bt, social_dists = _prepare_social_inputs(payouts=payouts, social_dists=social_dists)
    nblocks, ntrials = payouts_bt.shape[:2]
    choice_idx = _choices_to_idx(choices, nblocks=nblocks, ntrials=ntrials)

    params = np.asarray(params, dtype=float)
    if params.shape != (1,):
        raise ValueError("params must have shape (1,) in normalized space: [k]")

    k = float(norm2beta(params[0]))
    if not (k > 0.0):
        return 1e7

    U_self = np.zeros((nblocks, ntrials), dtype=float)
    U_other = np.zeros((nblocks, ntrials), dtype=float)
    EV = np.zeros((nblocks, ntrials), dtype=float)
    delta_V = np.zeros((nblocks, ntrials), dtype=float)
    p_prosocial = np.zeros((nblocks, ntrials), dtype=float)
    ch_prob = np.zeros((nblocks, ntrials, 2), dtype=float)
    outcomes = np.zeros((nblocks, ntrials, 2), dtype=float)
    choices_prosocial = np.zeros((nblocks, ntrials), dtype=float)
    nll_trialwise = np.zeros((nblocks, ntrials), dtype=float)
    nll = 0.0

    for b in range(nblocks):
        N = float(social_dists[b])

        for t in range(ntrials):
            trial_payouts = payouts_bt[b, t, :, :]
            r_self = trial_payouts[:, 0]
            r_other = trial_payouts[:, 1]

            U_self_opts = r_self.copy()
            U_other_opts = np.where(r_other > 0.0, r_other - k * (N ** 2), 0.0)
            V_opts = U_self_opts + U_other_opts

            dv = V_opts[1] - V_opts[0]
            p1 = float(expit(dv))
            p = np.array([1.0 - p1, p1])
            c = int(choice_idx[b, t])

            delta_V[b, t] = dv
            p_prosocial[b, t] = p1
            ch_prob[b, t, :] = p
            choices_prosocial[b, t] = float(c == 1)
            outcomes[b, t, :] = trial_payouts[c, :]
            U_self[b, t] = U_self_opts[c]
            U_other[b, t] = U_other_opts[c]
            EV[b, t] = V_opts[c]
            nll_trialwise[b, t] = -np.log(p[c] + 1e-12)
            nll += nll_trialwise[b, t]

    if output == "all":
        return _fit_output([k], payouts_bt, social_dists, U_self, U_other, EV, delta_V,
                            p_prosocial, choices, ch_prob, choices_prosocial, outcomes, nll, nll_trialwise)

    return calc_fval(nll, params, prior=prior, output=output)


sd_par_k_desc = """Parabolic social discounting: U_other(N) = r_other - k*N**2 (for the
option that actually offers the target something; 0 for the selfish option,
regardless of N; w_other fixed at 1). p(prosocial) = sigmoid(V_prosocial - V_selfish).
Because N**2 grows fast, k must be much smaller than in the linear model to
produce comparable discounting over the same N range.
Free parameter: k (parabolic discount rate, >0)."""
sd_par_k_id = "sd_par_k"
sd_par_k_spec = {"social_discounting": {"weight": [], "discount": ["k"]}, "shape": "parabolic", "choice_rule": "sigmoid(delta_V)"}
sd_par_k_model = ModelSpec(
    id=sd_par_k_id, spec=sd_par_k_spec, desc=sd_par_k_desc.strip(),
    params=None, sim=sd_par_k_sim, fit=sd_par_k_fit,
)


# =============================================================================
# sd_lin_k -- linear discounting, 1 free parameter (k; w_other fixed at 1)
#   U_other(N) = r_other - k*N   (only where r_other > 0)
# =============================================================================
def sd_lin_k_sim(params: np.ndarray, payouts=TASK1_PAYOUTS, social_dists=DEFAULT_SOCIAL_DISTS) -> dict:
    """Simulate the 1-parameter linear social discounting model (k free)."""
    payouts_bt, social_dists = _prepare_social_inputs(payouts=payouts, social_dists=social_dists)
    params = np.asarray(params, dtype=float)
    if params.ndim != 2 or params.shape[1] != 1:
        raise ValueError("params must have shape (nsubjects, 1): column [k]")

    nsubjects = params.shape[0]
    nblocks, ntrials = payouts_bt.shape[:2]
    payouts_subj, social_dists_subj = _expand_for_subjects(payouts_bt, social_dists, nsubjects)
    rng = np.random.default_rng()

    choices = np.empty((nsubjects, nblocks, ntrials), dtype=object)
    choices_prosocial = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    ch_prob = np.zeros((nsubjects, nblocks, ntrials, 2), dtype=float)
    p_prosocial = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    outcomes = np.zeros((nsubjects, nblocks, ntrials, 2), dtype=float)
    U_self = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    U_other = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    EV = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    delta_V = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    nll = np.zeros((nsubjects, nblocks, ntrials), dtype=float)

    all_k = params[:, 0]
    if not (all_k > 0.0).all():
        raise ValueError("k must be > 0")

    for s in range(nsubjects):
        k = float(all_k[s])

        for b in range(nblocks):
            N = float(social_dists[b])

            for t in range(ntrials):
                trial_payouts = payouts_bt[b, t, :, :]
                r_self = trial_payouts[:, 0]
                r_other = trial_payouts[:, 1]

                U_self_opts = r_self.copy()
                U_other_opts = np.where(r_other > 0.0, r_other - k * N, 0.0)
                V_opts = U_self_opts + U_other_opts

                dv = V_opts[1] - V_opts[0]
                p1 = float(expit(dv))
                p = np.array([1.0 - p1, p1])
                c = int(rng.choice([0, 1], p=p))

                ch_prob[s, b, t, :] = p
                p_prosocial[s, b, t] = p1
                delta_V[s, b, t] = dv
                choices[s, b, t] = CHOICE_LABELS[c]
                choices_prosocial[s, b, t] = float(c == 1)
                outcomes[s, b, t, :] = trial_payouts[c, :]
                U_self[s, b, t] = U_self_opts[c]
                U_other[s, b, t] = U_other_opts[c]
                EV[s, b, t] = V_opts[c]
                nll[s, b, t] = -np.log(p[c] + 1e-12)

    return _sim_output(params, payouts_subj, social_dists_subj, U_self, U_other, EV, delta_V,
                        p_prosocial, choices, ch_prob, choices_prosocial, outcomes, nll)


def sd_lin_k_fit(params, choices, payouts, social_dists, prior=None, output: str = "npl"):
    """Fit the 1-parameter linear social discounting model. params: (1,) Gaussian-space [k]."""
    payouts_bt, social_dists = _prepare_social_inputs(payouts=payouts, social_dists=social_dists)
    nblocks, ntrials = payouts_bt.shape[:2]
    choice_idx = _choices_to_idx(choices, nblocks=nblocks, ntrials=ntrials)

    params = np.asarray(params, dtype=float)
    if params.shape != (1,):
        raise ValueError("params must have shape (1,) in normalized space: [k]")

    k = float(norm2beta(params[0]))
    if not (k > 0.0):
        return 1e7

    U_self = np.zeros((nblocks, ntrials), dtype=float)
    U_other = np.zeros((nblocks, ntrials), dtype=float)
    EV = np.zeros((nblocks, ntrials), dtype=float)
    delta_V = np.zeros((nblocks, ntrials), dtype=float)
    p_prosocial = np.zeros((nblocks, ntrials), dtype=float)
    ch_prob = np.zeros((nblocks, ntrials, 2), dtype=float)
    outcomes = np.zeros((nblocks, ntrials, 2), dtype=float)
    choices_prosocial = np.zeros((nblocks, ntrials), dtype=float)
    nll_trialwise = np.zeros((nblocks, ntrials), dtype=float)
    nll = 0.0

    for b in range(nblocks):
        N = float(social_dists[b])

        for t in range(ntrials):
            trial_payouts = payouts_bt[b, t, :, :]
            r_self = trial_payouts[:, 0]
            r_other = trial_payouts[:, 1]

            U_self_opts = r_self.copy()
            U_other_opts = np.where(r_other > 0.0, r_other - k * N, 0.0)
            V_opts = U_self_opts + U_other_opts

            dv = V_opts[1] - V_opts[0]
            p1 = float(expit(dv))
            p = np.array([1.0 - p1, p1])
            c = int(choice_idx[b, t])

            delta_V[b, t] = dv
            p_prosocial[b, t] = p1
            ch_prob[b, t, :] = p
            choices_prosocial[b, t] = float(c == 1)
            outcomes[b, t, :] = trial_payouts[c, :]
            U_self[b, t] = U_self_opts[c]
            U_other[b, t] = U_other_opts[c]
            EV[b, t] = V_opts[c]
            nll_trialwise[b, t] = -np.log(p[c] + 1e-12)
            nll += nll_trialwise[b, t]

    if output == "all":
        return _fit_output([k], payouts_bt, social_dists, U_self, U_other, EV, delta_V,
                            p_prosocial, choices, ch_prob, choices_prosocial, outcomes, nll, nll_trialwise)

    return calc_fval(nll, params, prior=prior, output=output)


sd_lin_k_desc = """Linear social discounting: U_other(N) = r_other - k*N (for the option
that actually offers the target something; 0 for the selfish option,
regardless of N; w_other fixed at 1). p(prosocial) = sigmoid(V_prosocial - V_selfish).
Utility is not floored at zero, so heavily-discounted prosocial offers can
carry negative utility at large N.
Free parameter: k (linear discount rate, >0)."""
sd_lin_k_id = "sd_lin_k"
sd_lin_k_spec = {"social_discounting": {"weight": [], "discount": ["k"]}, "shape": "linear", "choice_rule": "sigmoid(delta_V)"}
sd_lin_k_model = ModelSpec(
    id=sd_lin_k_id, spec=sd_lin_k_spec, desc=sd_lin_k_desc.strip(),
    params=None, sim=sd_lin_k_sim, fit=sd_lin_k_fit,
)
