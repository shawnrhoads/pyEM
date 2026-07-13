"""discounting over binary two-option choice tasks.

Each sub-family below models a different way that the subjective value of a
reward gets discounted -- by social distance, delay, probability, or
required effort -- using the same generative shape: a binary choice between
a "baseline" option (undiscounted) and a "discounted" option whose value
shrinks as a function of a block-level discounting variable:

    Social discounting            (sd_*)   -- distance to a social target
    Temporal (delay) discounting  (td_*)   -- delay until a larger reward
    Probability discounting       (prd_*)  -- odds against a risky reward
    Effort discounting            (ed_*)   -- physical/cognitive effort cost
    Prosocial effort discounting  (ped_*)  -- effort exerted for self vs other

As the block-level discounting variable grows, the discounted option's value falls
and the indifference point sweeps down the ladder, so the trial at which the
subject switches between options is informative about k at every block level.
(If instead the discounted option were the *smaller* reward, it would almost
never be chosen and the data would carry almost no information about k.)

The probability of choosing the "discounted" option is a
logistic (sigmoid) function of the value difference:

    p(discounted option) = sigmoid(V_discounted - V_baseline)

"""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from ..core.modelspec import ModelSpec
from ..utils.math import norm2beta, calc_fval

# =============================================================================
# Shared helpers
# =============================================================================
def _choices_to_idx(choices: np.ndarray, nblocks: int, ntrials: int, choice_map: dict | None = None) -> np.ndarray:
    """Map a (nblocks, ntrials) array of choice labels or 0/1 codes to indices."""
    arr = np.asarray(choices)
    if arr.shape != (nblocks, ntrials):
        raise ValueError("choices must have shape (nblocks, ntrials)")

    if np.issubdtype(arr.dtype, np.number):
        idx = arr.astype(int, copy=False)
        if not np.isin(idx, [0, 1]).all():
            raise ValueError("Numeric choices must be coded as 0 or 1")
        return idx

    if choice_map is None:
        choice_map = CHOICE_MAP

    idx = np.empty((nblocks, ntrials), dtype=int)
    for b in range(nblocks):
        for t in range(ntrials):
            key = arr[b, t]
            if isinstance(key, str):
                key = key.strip().lower()
            if key not in choice_map:
                raise ValueError(f"choices must be coded via {sorted(set(choice_map))} or numeric 0/1")
            idx[b, t] = choice_map[key]
    return idx


def _prepare_target_inputs(
    payouts: np.ndarray | None,
    block_var: np.ndarray | None,
    default_payouts: np.ndarray,
    default_block_var: np.ndarray,
    var_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse (payouts, block_var) for the (..., 2, 2) self/other payout shape
    used by the social-discounting task."""
    if block_var is None:
        block_var = default_block_var.copy()
    block_var = np.asarray(block_var, dtype=float)

    if block_var.ndim != 1:
        raise ValueError(f"{var_name} must be a 1D array")
    if block_var.size == 0:
        raise ValueError(f"{var_name} must contain at least one block value")
    if np.any(block_var <= 0.0):
        raise ValueError(f"{var_name} values must be > 0")

    if payouts is None:
        payouts = default_payouts.copy()
    payouts = np.asarray(payouts, dtype=float)

    if payouts.ndim == 3:
        if payouts.shape[1:] != (2, 2):
            raise ValueError("3D payouts must have shape (ntrials, 2, 2)")
        nblocks = block_var.size
        ntrials = payouts.shape[0]
        payouts_bt = np.broadcast_to(payouts[None, ...], (nblocks, ntrials, 2, 2)).copy()
    elif payouts.ndim == 4:
        if payouts.shape[2:] != (2, 2):
            raise ValueError("4D payouts must have shape (nblocks, ntrials, 2, 2)")
        if payouts.shape[0] == 1:
            payouts_bt = np.broadcast_to(payouts, (block_var.size, payouts.shape[1], 2, 2)).copy()
        elif payouts.shape[0] == block_var.size:
            payouts_bt = payouts.copy()
        else:
            raise ValueError(f"If payouts is 4D, its first dimension must equal len({var_name}) or be 1")
    else:
        raise ValueError("payouts must have shape (ntrials, 2, 2) or (nblocks, ntrials, 2, 2)")

    return payouts_bt, block_var


def _expand_target_for_subjects(payouts_bt: np.ndarray, block_var: np.ndarray, nsubjects: int) -> tuple[np.ndarray, np.ndarray]:
    nblocks, ntrials = payouts_bt.shape[:2]
    payouts_subj = np.broadcast_to(payouts_bt[None, ...], (nsubjects, nblocks, ntrials, 2, 2)).copy()
    block_var_subj = np.broadcast_to(block_var[None, :], (nsubjects, nblocks)).copy()
    return payouts_subj, block_var_subj


def _prepare_2opt_inputs(
    payouts: np.ndarray | None,
    block_var: np.ndarray | None,
    default_payouts: np.ndarray,
    default_block_var: np.ndarray,
    var_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse (payouts, block_var) for the (..., 2) two-option payout shape
    used by the temporal/probability/effort discounting tasks (no self/other
    split -- just one amount per option)."""
    if block_var is None:
        block_var = default_block_var.copy()
    block_var = np.asarray(block_var, dtype=float)

    if block_var.ndim != 1:
        raise ValueError(f"{var_name} must be a 1D array")
    if block_var.size == 0:
        raise ValueError(f"{var_name} must contain at least one block value")
    if np.any(block_var <= 0.0):
        raise ValueError(f"{var_name} values must be > 0")

    if payouts is None:
        payouts = default_payouts.copy()
    payouts = np.asarray(payouts, dtype=float)

    if payouts.ndim == 2:
        if payouts.shape[1] != 2:
            raise ValueError("2D payouts must have shape (ntrials, 2)")
        nblocks = block_var.size
        ntrials = payouts.shape[0]
        payouts_bt = np.broadcast_to(payouts[None, ...], (nblocks, ntrials, 2)).copy()
    elif payouts.ndim == 3:
        if payouts.shape[2] != 2:
            raise ValueError("3D payouts must have shape (nblocks, ntrials, 2)")
        if payouts.shape[0] == 1:
            payouts_bt = np.broadcast_to(payouts, (block_var.size, payouts.shape[1], 2)).copy()
        elif payouts.shape[0] == block_var.size:
            payouts_bt = payouts.copy()
        else:
            raise ValueError(f"If payouts is 3D, its first dimension must equal len({var_name}) or be 1")
    else:
        raise ValueError("payouts must have shape (ntrials, 2) or (nblocks, ntrials, 2)")

    return payouts_bt, block_var


def _expand_2opt_for_subjects(payouts_bt: np.ndarray, block_var: np.ndarray, nsubjects: int) -> tuple[np.ndarray, np.ndarray]:
    nblocks, ntrials = payouts_bt.shape[:2]
    payouts_subj = np.broadcast_to(payouts_bt[None, ...], (nsubjects, nblocks, ntrials, 2)).copy()
    block_var_subj = np.broadcast_to(block_var[None, :], (nsubjects, nblocks)).copy()
    return payouts_subj, block_var_subj


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


# #############################################################################
# Social discounting
#
# On each trial, an agent chooses between:
#   - a "selfish" option: keep everything, give the social target nothing.
#   - a "prosocial" option: split the pot, with the target's share landing
#     at social distance N (1 = closest, 100 = a stranger).
#
# Four model variants (sd_hyp_wk, sd_hyp_k, sd_par_k, sd_lin_k) share this
# task structure; sd_hyp_wk has a free other-regarding weight, the rest fix
# it at 1. See each ModelSpec's `desc` for its exact value function.
#
# Axis convention for payouts: payouts[trial, choice, target] where
# choice 0='selfish', 1='prosocial' and target 0='self', 1='other'.
# #############################################################################
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

# TASK2_PAYOUTS = np.array([
#     [[ 95.0,   0.0], [  0.0, 105.0]],
#     [[ 85.0,   0.0], [  0.0, 105.0]],
#     [[ 75.0,   0.0], [  0.0, 105.0]],
#     [[ 65.0,   0.0], [  0.0, 105.0]],
#     [[ 55.0,   0.0], [  0.0, 105.0]],
#     [[ 45.0,   0.0], [  0.0, 105.0]],
#     [[ 35.0,   0.0], [  0.0, 105.0]],
#     [[ 25.0,   0.0], [  0.0, 105.0]],
#     [[ 15.0,   0.0], [  0.0, 105.0]],
# ], dtype=float)

CHOICE_LABELS = np.array(["selfish", "prosocial"], dtype=object)
CHOICE_MAP = {
    "selfish": 0, "prosocial": 1,
    "self": 0, "other": 1,
    "s": 0, "p": 1,
    0: 0, 1: 1,
}


# =============================================================================
# sd_hyp_wk -- hyperbolic discounting, 2 free parameters (w_other, k)
#   U_other(N) = w_other * r_other / (1 + k*N)
# =============================================================================
def sd_hyp_wk_sim(params: np.ndarray, payouts=TASK1_PAYOUTS, social_dists=DEFAULT_SOCIAL_DISTS, seed: int | None = None) -> dict:
    """Simulate the 2-parameter hyperbolic social discounting model (w_other, k free)."""
    payouts_bt, social_dists = _prepare_target_inputs(payouts, social_dists, TASK1_PAYOUTS, DEFAULT_SOCIAL_DISTS, "social_dists")
    params = np.asarray(params, dtype=float)
    if params.ndim != 2 or params.shape[1] != 2:
        raise ValueError("params must have shape (nsubjects, 2): columns [w_other, k]")

    nsubjects = params.shape[0]
    nblocks, ntrials = payouts_bt.shape[:2]
    payouts_subj, social_dists_subj = _expand_target_for_subjects(payouts_bt, social_dists, nsubjects)
    rng = np.random.default_rng(seed)

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
    payouts_bt, social_dists = _prepare_target_inputs(payouts, social_dists, TASK1_PAYOUTS, DEFAULT_SOCIAL_DISTS, "social_dists")
    nblocks, ntrials = payouts_bt.shape[:2]
    choice_idx = _choices_to_idx(choices, nblocks=nblocks, ntrials=ntrials, choice_map=CHOICE_MAP)

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
def sd_hyp_k_sim(params: np.ndarray, payouts=TASK1_PAYOUTS, social_dists=DEFAULT_SOCIAL_DISTS, seed: int | None = None) -> dict:
    """Simulate the 1-parameter hyperbolic social discounting model (k free)."""
    payouts_bt, social_dists = _prepare_target_inputs(payouts, social_dists, TASK1_PAYOUTS, DEFAULT_SOCIAL_DISTS, "social_dists")
    params = np.asarray(params, dtype=float)
    if params.ndim != 2 or params.shape[1] != 1:
        raise ValueError("params must have shape (nsubjects, 1): column [k]")

    nsubjects = params.shape[0]
    nblocks, ntrials = payouts_bt.shape[:2]
    payouts_subj, social_dists_subj = _expand_target_for_subjects(payouts_bt, social_dists, nsubjects)
    rng = np.random.default_rng(seed)

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
    payouts_bt, social_dists = _prepare_target_inputs(payouts, social_dists, TASK1_PAYOUTS, DEFAULT_SOCIAL_DISTS, "social_dists")
    nblocks, ntrials = payouts_bt.shape[:2]
    choice_idx = _choices_to_idx(choices, nblocks=nblocks, ntrials=ntrials, choice_map=CHOICE_MAP)

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
def sd_par_k_sim(params: np.ndarray, payouts=TASK1_PAYOUTS, social_dists=DEFAULT_SOCIAL_DISTS, seed: int | None = None) -> dict:
    """Simulate the 1-parameter parabolic social discounting model (k free)."""
    payouts_bt, social_dists = _prepare_target_inputs(payouts, social_dists, TASK1_PAYOUTS, DEFAULT_SOCIAL_DISTS, "social_dists")
    params = np.asarray(params, dtype=float)
    if params.ndim != 2 or params.shape[1] != 1:
        raise ValueError("params must have shape (nsubjects, 1): column [k]")

    nsubjects = params.shape[0]
    nblocks, ntrials = payouts_bt.shape[:2]
    payouts_subj, social_dists_subj = _expand_target_for_subjects(payouts_bt, social_dists, nsubjects)
    rng = np.random.default_rng(seed)

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
    payouts_bt, social_dists = _prepare_target_inputs(payouts, social_dists, TASK1_PAYOUTS, DEFAULT_SOCIAL_DISTS, "social_dists")
    nblocks, ntrials = payouts_bt.shape[:2]
    choice_idx = _choices_to_idx(choices, nblocks=nblocks, ntrials=ntrials, choice_map=CHOICE_MAP)

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
def sd_lin_k_sim(params: np.ndarray, payouts=TASK1_PAYOUTS, social_dists=DEFAULT_SOCIAL_DISTS, seed: int | None = None) -> dict:
    """Simulate the 1-parameter linear social discounting model (k free)."""
    payouts_bt, social_dists = _prepare_target_inputs(payouts, social_dists, TASK1_PAYOUTS, DEFAULT_SOCIAL_DISTS, "social_dists")
    params = np.asarray(params, dtype=float)
    if params.ndim != 2 or params.shape[1] != 1:
        raise ValueError("params must have shape (nsubjects, 1): column [k]")

    nsubjects = params.shape[0]
    nblocks, ntrials = payouts_bt.shape[:2]
    payouts_subj, social_dists_subj = _expand_target_for_subjects(payouts_bt, social_dists, nsubjects)
    rng = np.random.default_rng(seed)

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
    payouts_bt, social_dists = _prepare_target_inputs(payouts, social_dists, TASK1_PAYOUTS, DEFAULT_SOCIAL_DISTS, "social_dists")
    nblocks, ntrials = payouts_bt.shape[:2]
    choice_idx = _choices_to_idx(choices, nblocks=nblocks, ntrials=ntrials, choice_map=CHOICE_MAP)

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


# #############################################################################
# Temporal (delay) discounting
#
# On each trial, an agent chooses between:
#   - "sooner": a SMALLER amount available immediately (undiscounted).
#   - "later":  a LARGER, fixed amount available after delay D (block-level,
#     1 = shortest wait, largest D = longest wait), discounted by the wait.
#
# This is the standard smaller-sooner vs larger-later paradigm. Only the
# later reward is delay-discounted; the immediate (sooner) reward is not.
# The sooner amount steps down a 9-rung ladder so the indifference point
# 155/(1+k*D) sweeps the ladder as D grows -- which is what identifies k.
#
# Axis convention for payouts: payouts[trial, choice] where choice
# 0='sooner', 1='later' -- a single amount per option.
# #############################################################################
DEFAULT_DELAYS = np.array([1, 2, 5, 10, 20, 50, 100], dtype=float)  # delay units (e.g. days)

TD_PAYOUTS = np.array([
    [145.0, 155.0],
    [130.0, 155.0],
    [115.0, 155.0],
    [100.0, 155.0],
    [ 85.0, 155.0],
    [ 70.0, 155.0],
    [ 55.0, 155.0],
    [ 40.0, 155.0],
    [ 25.0, 155.0],
], dtype=float)  # [sooner (smaller, immediate), later (larger, delayed)]

CHOICE_LABELS_TD = np.array(["sooner", "later"], dtype=object)
CHOICE_MAP_TD = {
    "sooner": 0, "later": 1,
    "ss": 0, "ll": 1,
    "immediate": 0, "delayed": 1,
    0: 0, 1: 1,
}


def td_hyp_k_sim(params: np.ndarray, payouts=TD_PAYOUTS, delays=DEFAULT_DELAYS, seed: int | None = None) -> dict:
    """Simulate the 1-parameter hyperbolic temporal discounting model (k free)."""
    payouts_bt, delays = _prepare_2opt_inputs(payouts, delays, TD_PAYOUTS, DEFAULT_DELAYS, "delays")
    params = np.asarray(params, dtype=float)
    if params.ndim != 2 or params.shape[1] != 1:
        raise ValueError("params must have shape (nsubjects, 1): column [k]")

    nsubjects = params.shape[0]
    nblocks, ntrials = payouts_bt.shape[:2]
    payouts_subj, delays_subj = _expand_2opt_for_subjects(payouts_bt, delays, nsubjects)
    rng = np.random.default_rng(seed)

    choices = np.empty((nsubjects, nblocks, ntrials), dtype=object)
    choices_later = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    ch_prob = np.zeros((nsubjects, nblocks, ntrials, 2), dtype=float)
    p_later = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    outcomes = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    V_sooner = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    V_later = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    EV = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    delta_V = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    nll = np.zeros((nsubjects, nblocks, ntrials), dtype=float)

    all_k = params[:, 0]
    if not (all_k > 0.0).all():
        raise ValueError("k must be > 0")

    for s in range(nsubjects):
        k = float(all_k[s])

        for b in range(nblocks):
            D = float(delays[b])

            for t in range(ntrials):
                r_sooner = payouts_bt[b, t, 0]   # smaller, immediate
                r_later = payouts_bt[b, t, 1]    # larger, delayed

                V_sooner_val = r_sooner                    # immediate -> undiscounted
                V_later_val = r_later / (1.0 + k * D)      # delayed -> discounted

                dv = V_later_val - V_sooner_val
                p1 = float(expit(dv))
                p = np.array([1.0 - p1, p1])
                c = int(rng.choice([0, 1], p=p))

                ch_prob[s, b, t, :] = p
                p_later[s, b, t] = p1
                delta_V[s, b, t] = dv
                choices[s, b, t] = CHOICE_LABELS_TD[c]
                choices_later[s, b, t] = float(c == 1)
                outcomes[s, b, t] = payouts_bt[b, t, c]
                V_sooner[s, b, t] = V_sooner_val
                V_later[s, b, t] = V_later_val
                EV[s, b, t] = V_later_val if c == 1 else V_sooner_val
                nll[s, b, t] = -np.log(p[c] + 1e-12)

    return {
        "params": params.copy(),
        "payouts": payouts_subj,
        "delays": delays_subj,
        "V_sooner": V_sooner,
        "V_later": V_later,
        "EV": EV,
        "delta_V": delta_V,
        "p_later": p_later,
        "choices": choices,
        "ch_prob": ch_prob,
        "choices_later": choices_later,
        "outcomes": outcomes,
        "nll": nll,
        "nll_total": nll.sum(axis=(1, 2)),
    }


def td_hyp_k_fit(params, choices, payouts, delays, prior=None, output: str = "npl"):
    """Fit the 1-parameter hyperbolic temporal discounting model. params: (1,) Gaussian-space [k]."""
    payouts_bt, delays = _prepare_2opt_inputs(payouts, delays, TD_PAYOUTS, DEFAULT_DELAYS, "delays")
    nblocks, ntrials = payouts_bt.shape[:2]
    choice_idx = _choices_to_idx(choices, nblocks=nblocks, ntrials=ntrials, choice_map=CHOICE_MAP_TD)

    params = np.asarray(params, dtype=float)
    if params.shape != (1,):
        raise ValueError("params must have shape (1,) in normalized space: [k]")

    k = float(norm2beta(params[0]))
    if not (k > 0.0):
        return 1e7

    V_sooner = np.zeros((nblocks, ntrials), dtype=float)
    V_later = np.zeros((nblocks, ntrials), dtype=float)
    EV = np.zeros((nblocks, ntrials), dtype=float)
    delta_V = np.zeros((nblocks, ntrials), dtype=float)
    p_later = np.zeros((nblocks, ntrials), dtype=float)
    ch_prob = np.zeros((nblocks, ntrials, 2), dtype=float)
    outcomes = np.zeros((nblocks, ntrials), dtype=float)
    choices_later = np.zeros((nblocks, ntrials), dtype=float)
    nll_trialwise = np.zeros((nblocks, ntrials), dtype=float)
    nll = 0.0

    for b in range(nblocks):
        D = float(delays[b])

        for t in range(ntrials):
            r_sooner = payouts_bt[b, t, 0]
            r_later = payouts_bt[b, t, 1]

            V_sooner_val = r_sooner
            V_later_val = r_later / (1.0 + k * D)

            dv = V_later_val - V_sooner_val
            p1 = float(expit(dv))
            p = np.array([1.0 - p1, p1])
            c = int(choice_idx[b, t])

            delta_V[b, t] = dv
            p_later[b, t] = p1
            ch_prob[b, t, :] = p
            choices_later[b, t] = float(c == 1)
            outcomes[b, t] = payouts_bt[b, t, c]
            V_sooner[b, t] = V_sooner_val
            V_later[b, t] = V_later_val
            EV[b, t] = V_later_val if c == 1 else V_sooner_val
            nll_trialwise[b, t] = -np.log(p[c] + 1e-12)
            nll += nll_trialwise[b, t]

    if output == "all":
        return {
            "params": [k],
            "payouts": payouts_bt,
            "delays": delays.copy(),
            "V_sooner": V_sooner,
            "V_later": V_later,
            "EV": EV,
            "delta_V": delta_V,
            "p_later": p_later,
            "choices": np.asarray(choices, dtype=object).copy(),
            "ch_prob": ch_prob,
            "choices_later": choices_later,
            "outcomes": outcomes,
            "nll": nll,
            "nll_trialwise": nll_trialwise,
        }

    return calc_fval(nll, params, prior=prior, output=output)


td_hyp_k_desc = """Hyperbolic temporal (delay) discounting (Mazur, 1987) in a
smaller-sooner vs larger-later task: V_later(D) = r_later / (1 + k*D), while
the smaller immediate 'sooner' reward is undiscounted (V_sooner = r_sooner).
p(later) = sigmoid(V_later - V_sooner).
Free parameter: k (delay discount rate, >0)."""
td_hyp_k_id = "td_hyp_k"
td_hyp_k_spec = {"temporal_discounting": {"weight": [], "discount": ["k"]}, "shape": "hyperbolic", "choice_rule": "sigmoid(delta_V)"}
td_hyp_k_model = ModelSpec(
    id=td_hyp_k_id, spec=td_hyp_k_spec, desc=td_hyp_k_desc.strip(),
    params=None, sim=td_hyp_k_sim, fit=td_hyp_k_fit,
)


# #############################################################################
# Probability discounting
#
# On each trial, an agent chooses between:
#   - "certain": a SMALLER amount received with probability 1 (undiscounted).
#   - "risky":   a LARGER, fixed amount received only with probability p
#     (block-level; smaller p = riskier), discounted by the odds against.
#
# Standard certain-small vs risky-large paradigm. Only the risky reward is
# probability-discounted, via the odds against winning theta = (1-p)/p. The
# certain amount steps down a 9-rung ladder so the indifference point sweeps
# the ladder as p falls -- which is what identifies k.
#
# Axis convention for payouts: payouts[trial, choice] where choice
# 0='certain', 1='risky' -- a single amount per option.
# #############################################################################
DEFAULT_PROBS = np.array([0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05], dtype=float)

PRD_PAYOUTS = np.array([
    [145.0, 155.0],
    [130.0, 155.0],
    [115.0, 155.0],
    [100.0, 155.0],
    [ 85.0, 155.0],
    [ 70.0, 155.0],
    [ 55.0, 155.0],
    [ 40.0, 155.0],
    [ 25.0, 155.0],
], dtype=float)  # [certain (smaller, sure), risky (larger, probabilistic)]

CHOICE_LABELS_PRD = np.array(["certain", "risky"], dtype=object)
CHOICE_MAP_PRD = {
    "certain": 0, "risky": 1,
    "safe": 0, "gamble": 1,
    0: 0, 1: 1,
}


def prd_hyp_k_sim(params: np.ndarray, payouts=PRD_PAYOUTS, probs=DEFAULT_PROBS, seed: int | None = None) -> dict:
    """Simulate the 1-parameter hyperbolic probability discounting model (k free)."""
    payouts_bt, probs = _prepare_2opt_inputs(payouts, probs, PRD_PAYOUTS, DEFAULT_PROBS, "probs")
    if np.any(probs > 1.0):
        raise ValueError("probs values must be in (0, 1]")
    params = np.asarray(params, dtype=float)
    if params.ndim != 2 or params.shape[1] != 1:
        raise ValueError("params must have shape (nsubjects, 1): column [k]")

    nsubjects = params.shape[0]
    nblocks, ntrials = payouts_bt.shape[:2]
    payouts_subj, probs_subj = _expand_2opt_for_subjects(payouts_bt, probs, nsubjects)
    rng = np.random.default_rng(seed)

    choices = np.empty((nsubjects, nblocks, ntrials), dtype=object)
    choices_risky = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    ch_prob = np.zeros((nsubjects, nblocks, ntrials, 2), dtype=float)
    p_risky = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    outcomes = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    V_certain = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    V_risky = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    EV = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    delta_V = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    nll = np.zeros((nsubjects, nblocks, ntrials), dtype=float)

    all_k = params[:, 0]
    if not (all_k > 0.0).all():
        raise ValueError("k must be > 0")

    for s in range(nsubjects):
        k = float(all_k[s])

        for b in range(nblocks):
            p_win = float(probs[b])
            theta = (1.0 - p_win) / p_win   # odds against winning

            for t in range(ntrials):
                r_certain = payouts_bt[b, t, 0]  # smaller, sure
                r_risky = payouts_bt[b, t, 1]    # larger, probabilistic

                V_certain_val = r_certain                  # sure thing -> undiscounted
                V_risky_val = r_risky / (1.0 + k * theta)  # discounted by odds against

                dv = V_risky_val - V_certain_val
                p1 = float(expit(dv))
                p = np.array([1.0 - p1, p1])
                c = int(rng.choice([0, 1], p=p))

                ch_prob[s, b, t, :] = p
                p_risky[s, b, t] = p1
                delta_V[s, b, t] = dv
                choices[s, b, t] = CHOICE_LABELS_PRD[c]
                choices_risky[s, b, t] = float(c == 1)
                outcomes[s, b, t] = payouts_bt[b, t, c]
                V_certain[s, b, t] = V_certain_val
                V_risky[s, b, t] = V_risky_val
                EV[s, b, t] = V_risky_val if c == 1 else V_certain_val
                nll[s, b, t] = -np.log(p[c] + 1e-12)

    return {
        "params": params.copy(),
        "payouts": payouts_subj,
        "probs": probs_subj,
        "V_certain": V_certain,
        "V_risky": V_risky,
        "EV": EV,
        "delta_V": delta_V,
        "p_risky": p_risky,
        "choices": choices,
        "ch_prob": ch_prob,
        "choices_risky": choices_risky,
        "outcomes": outcomes,
        "nll": nll,
        "nll_total": nll.sum(axis=(1, 2)),
    }


def prd_hyp_k_fit(params, choices, payouts, probs, prior=None, output: str = "npl"):
    """Fit the 1-parameter hyperbolic probability discounting model. params: (1,) Gaussian-space [k]."""
    payouts_bt, probs = _prepare_2opt_inputs(payouts, probs, PRD_PAYOUTS, DEFAULT_PROBS, "probs")
    if np.any(probs > 1.0):
        raise ValueError("probs values must be in (0, 1]")
    nblocks, ntrials = payouts_bt.shape[:2]
    choice_idx = _choices_to_idx(choices, nblocks=nblocks, ntrials=ntrials, choice_map=CHOICE_MAP_PRD)

    params = np.asarray(params, dtype=float)
    if params.shape != (1,):
        raise ValueError("params must have shape (1,) in normalized space: [k]")

    k = float(norm2beta(params[0]))
    if not (k > 0.0):
        return 1e7

    V_certain = np.zeros((nblocks, ntrials), dtype=float)
    V_risky = np.zeros((nblocks, ntrials), dtype=float)
    EV = np.zeros((nblocks, ntrials), dtype=float)
    delta_V = np.zeros((nblocks, ntrials), dtype=float)
    p_risky = np.zeros((nblocks, ntrials), dtype=float)
    ch_prob = np.zeros((nblocks, ntrials, 2), dtype=float)
    outcomes = np.zeros((nblocks, ntrials), dtype=float)
    choices_risky = np.zeros((nblocks, ntrials), dtype=float)
    nll_trialwise = np.zeros((nblocks, ntrials), dtype=float)
    nll = 0.0

    for b in range(nblocks):
        p_win = float(probs[b])
        theta = (1.0 - p_win) / p_win

        for t in range(ntrials):
            r_certain = payouts_bt[b, t, 0]
            r_risky = payouts_bt[b, t, 1]

            V_certain_val = r_certain
            V_risky_val = r_risky / (1.0 + k * theta)

            dv = V_risky_val - V_certain_val
            p1 = float(expit(dv))
            p = np.array([1.0 - p1, p1])
            c = int(choice_idx[b, t])

            delta_V[b, t] = dv
            p_risky[b, t] = p1
            ch_prob[b, t, :] = p
            choices_risky[b, t] = float(c == 1)
            outcomes[b, t] = payouts_bt[b, t, c]
            V_certain[b, t] = V_certain_val
            V_risky[b, t] = V_risky_val
            EV[b, t] = V_risky_val if c == 1 else V_certain_val
            nll_trialwise[b, t] = -np.log(p[c] + 1e-12)
            nll += nll_trialwise[b, t]

    if output == "all":
        return {
            "params": [k],
            "payouts": payouts_bt,
            "probs": probs.copy(),
            "V_certain": V_certain,
            "V_risky": V_risky,
            "EV": EV,
            "delta_V": delta_V,
            "p_risky": p_risky,
            "choices": np.asarray(choices, dtype=object).copy(),
            "ch_prob": ch_prob,
            "choices_risky": choices_risky,
            "outcomes": outcomes,
            "nll": nll,
            "nll_trialwise": nll_trialwise,
        }

    return calc_fval(nll, params, prior=prior, output=output)


prd_hyp_k_desc = """Hyperbolic probability discounting (Rachlin, Raineri, & Cross, 1991):
V_risky(p) = r_risky / (1 + k*theta), where theta = (1-p)/p is the odds
against winning; the smaller certain option is undiscounted.
p(risky) = sigmoid(V_risky - V_certain).
Free parameter: k (probability discount rate, >0)."""
prd_hyp_k_id = "prd_hyp_k"
prd_hyp_k_spec = {"probability_discounting": {"weight": [], "discount": ["k"]}, "shape": "hyperbolic", "choice_rule": "sigmoid(delta_V)"}
prd_hyp_k_model = ModelSpec(
    id=prd_hyp_k_id, spec=prd_hyp_k_spec, desc=prd_hyp_k_desc.strip(),
    params=None, sim=prd_hyp_k_sim, fit=prd_hyp_k_fit,
)


# #############################################################################
# Effort discounting  (PARABOLIC)
#
# On each trial, an agent chooses between:
#   - "low_effort":  a SMALLER amount for minimal effort (undiscounted).
#   - "high_effort": a LARGER, fixed amount requiring effort E (block-level,
#     1 = least effortful, largest E = most effortful), discounted by E**2.
#
# The larger reward's value is reduced parabolically by the required effort,
# V_high(E) = r_high - k*E**2, matching the accelerating effort-cost form
# commonly used in the effort-discounting literature. The low-effort amount
# steps down a 9-rung ladder so the indifference point sweeps the ladder as
# E grows -- which is what identifies k.
#
# Axis convention for payouts: payouts[trial, choice] where choice
# 0='low_effort', 1='high_effort' -- a single amount per option.
# #############################################################################
DEFAULT_EFFORT_LEVELS = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)  # arbitrary effort units

ED_PAYOUTS = np.array([
    [145.0, 155.0],
    [130.0, 155.0],
    [115.0, 155.0],
    [100.0, 155.0],
    [ 85.0, 155.0],
    [ 70.0, 155.0],
    [ 55.0, 155.0],
    [ 40.0, 155.0],
    [ 25.0, 155.0],
], dtype=float)  # [low_effort (smaller), high_effort (larger)]

CHOICE_LABELS_ED = np.array(["low_effort", "high_effort"], dtype=object)
CHOICE_MAP_ED = {
    "low_effort": 0, "high_effort": 1,
    "low": 0, "high": 1,
    "rest": 0, "work": 1,
    0: 0, 1: 1,
}


def ed_par_k_sim(params: np.ndarray, payouts=ED_PAYOUTS, effort_levels=DEFAULT_EFFORT_LEVELS, seed: int | None = None) -> dict:
    """Simulate the 1-parameter parabolic effort discounting model (k free)."""
    payouts_bt, effort_levels = _prepare_2opt_inputs(payouts, effort_levels, ED_PAYOUTS, DEFAULT_EFFORT_LEVELS, "effort_levels")
    params = np.asarray(params, dtype=float)
    if params.ndim != 2 or params.shape[1] != 1:
        raise ValueError("params must have shape (nsubjects, 1): column [k]")

    nsubjects = params.shape[0]
    nblocks, ntrials = payouts_bt.shape[:2]
    payouts_subj, effort_subj = _expand_2opt_for_subjects(payouts_bt, effort_levels, nsubjects)
    rng = np.random.default_rng(seed)

    choices = np.empty((nsubjects, nblocks, ntrials), dtype=object)
    choices_high = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    ch_prob = np.zeros((nsubjects, nblocks, ntrials, 2), dtype=float)
    p_high = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    outcomes = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    V_low = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    V_high = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    EV = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    delta_V = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    nll = np.zeros((nsubjects, nblocks, ntrials), dtype=float)

    all_k = params[:, 0]
    if not (all_k > 0.0).all():
        raise ValueError("k must be > 0")

    for s in range(nsubjects):
        k = float(all_k[s])

        for b in range(nblocks):
            E = float(effort_levels[b])

            for t in range(ntrials):
                r_low = payouts_bt[b, t, 0]   # smaller, low effort
                r_high = payouts_bt[b, t, 1]  # larger, high effort

                V_low_val = r_low                       # minimal effort -> undiscounted
                V_high_val = r_high - k * (E ** 2)      # discounted parabolically by effort

                dv = V_high_val - V_low_val
                p1 = float(expit(dv))
                p = np.array([1.0 - p1, p1])
                c = int(rng.choice([0, 1], p=p))

                ch_prob[s, b, t, :] = p
                p_high[s, b, t] = p1
                delta_V[s, b, t] = dv
                choices[s, b, t] = CHOICE_LABELS_ED[c]
                choices_high[s, b, t] = float(c == 1)
                outcomes[s, b, t] = payouts_bt[b, t, c]
                V_low[s, b, t] = V_low_val
                V_high[s, b, t] = V_high_val
                EV[s, b, t] = V_high_val if c == 1 else V_low_val
                nll[s, b, t] = -np.log(p[c] + 1e-12)

    return {
        "params": params.copy(),
        "payouts": payouts_subj,
        "effort_levels": effort_subj,
        "V_low": V_low,
        "V_high": V_high,
        "EV": EV,
        "delta_V": delta_V,
        "p_high": p_high,
        "choices": choices,
        "ch_prob": ch_prob,
        "choices_high": choices_high,
        "outcomes": outcomes,
        "nll": nll,
        "nll_total": nll.sum(axis=(1, 2)),
    }


def ed_par_k_fit(params, choices, payouts, effort_levels, prior=None, output: str = "npl"):
    """Fit the 1-parameter parabolic effort discounting model. params: (1,) Gaussian-space [k]."""
    payouts_bt, effort_levels = _prepare_2opt_inputs(payouts, effort_levels, ED_PAYOUTS, DEFAULT_EFFORT_LEVELS, "effort_levels")
    nblocks, ntrials = payouts_bt.shape[:2]
    choice_idx = _choices_to_idx(choices, nblocks=nblocks, ntrials=ntrials, choice_map=CHOICE_MAP_ED)

    params = np.asarray(params, dtype=float)
    if params.shape != (1,):
        raise ValueError("params must have shape (1,) in normalized space: [k]")

    k = float(norm2beta(params[0]))
    if not (k > 0.0):
        return 1e7

    V_low = np.zeros((nblocks, ntrials), dtype=float)
    V_high = np.zeros((nblocks, ntrials), dtype=float)
    EV = np.zeros((nblocks, ntrials), dtype=float)
    delta_V = np.zeros((nblocks, ntrials), dtype=float)
    p_high = np.zeros((nblocks, ntrials), dtype=float)
    ch_prob = np.zeros((nblocks, ntrials, 2), dtype=float)
    outcomes = np.zeros((nblocks, ntrials), dtype=float)
    choices_high = np.zeros((nblocks, ntrials), dtype=float)
    nll_trialwise = np.zeros((nblocks, ntrials), dtype=float)
    nll = 0.0

    for b in range(nblocks):
        E = float(effort_levels[b])

        for t in range(ntrials):
            r_low = payouts_bt[b, t, 0]
            r_high = payouts_bt[b, t, 1]

            V_low_val = r_low
            V_high_val = r_high - k * (E ** 2)

            dv = V_high_val - V_low_val
            p1 = float(expit(dv))
            p = np.array([1.0 - p1, p1])
            c = int(choice_idx[b, t])

            delta_V[b, t] = dv
            p_high[b, t] = p1
            ch_prob[b, t, :] = p
            choices_high[b, t] = float(c == 1)
            outcomes[b, t] = payouts_bt[b, t, c]
            V_low[b, t] = V_low_val
            V_high[b, t] = V_high_val
            EV[b, t] = V_high_val if c == 1 else V_low_val
            nll_trialwise[b, t] = -np.log(p[c] + 1e-12)
            nll += nll_trialwise[b, t]

    if output == "all":
        return {
            "params": [k],
            "payouts": payouts_bt,
            "effort_levels": effort_levels.copy(),
            "V_low": V_low,
            "V_high": V_high,
            "EV": EV,
            "delta_V": delta_V,
            "p_high": p_high,
            "choices": np.asarray(choices, dtype=object).copy(),
            "ch_prob": ch_prob,
            "choices_high": choices_high,
            "outcomes": outcomes,
            "nll": nll,
            "nll_trialwise": nll_trialwise,
        }

    return calc_fval(nll, params, prior=prior, output=output)


ed_par_k_desc = """Parabolic effort discounting (accelerating effort cost, cf. Prevost,
Pessiglione, Metereau, Clery-Melin, & Dreher, 2010; Hartmann et al., 2013):
V_high(E) = r_high - k*E**2; the smaller low-effort option is undiscounted.
p(high_effort) = sigmoid(V_high - V_low).
Free parameter: k (effort discount rate, >0)."""
ed_par_k_id = "ed_par_k"
ed_par_k_spec = {"effort_discounting": {"weight": [], "discount": ["k"]}, "shape": "parabolic", "choice_rule": "sigmoid(delta_V)"}
ed_par_k_model = ModelSpec(
    id=ed_par_k_id, spec=ed_par_k_spec, desc=ed_par_k_desc.strip(),
    params=None, sim=ed_par_k_sim, fit=ed_par_k_fit,
)


# #############################################################################
# Prosocial effort discounting  (PARABOLIC)
#
# On each trial, an agent chooses between:
#   - "low_effort":  a SMALLER reward for the block's beneficiary, minimal
#     effort (undiscounted).
#   - "high_effort": a LARGER, fixed reward for the beneficiary, requiring
#     effort E (block-level), discounted parabolically by the effort.
#
# Blocks come in two types, indexed by a per-block beneficiary tag:
#     beneficiary 0 = "self"  -> reward goes to the chooser
#     beneficiary 1 = "other" -> reward goes to a social target
# The chooser always exerts the effort; only the recipient of the reward
# differs. This is the self/other effort paradigm of Lockwood et al. (2017),
# where willingness to work is typically discounted more steeply when the
# reward benefits someone else.
#
# Two models share this task:
#   ped_par_k   : 1 free parameter k, shared across self and other blocks.
#   ped_par_2k  : 2 free parameters [k_self, k_other], one effort discount
#                 rate per beneficiary type.
#
# Axis convention for payouts: payouts[trial, choice] where choice
# 0='low_effort', 1='high_effort' -- a single reward amount per option (paid
# to the block's beneficiary). Beneficiary is a separate per-block array.
# #############################################################################
PED_PAYOUTS = np.array([
    [145.0, 155.0],
    [130.0, 155.0],
    [115.0, 155.0],
    [100.0, 155.0],
    [ 85.0, 155.0],
    [ 70.0, 155.0],
    [ 55.0, 155.0],
    [ 40.0, 155.0],
    [ 25.0, 155.0],
], dtype=float)  # [low_effort (smaller), high_effort (larger)]

# 14 blocks: the 7 effort levels once for self-benefit, once for other-benefit.
PED_EFFORT_LEVELS = np.array([1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7], dtype=float)
PED_BENEFICIARY = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=float)  # 0=self, 1=other

BENEFICIARY_LABELS = np.array(["self", "other"], dtype=object)


def _prepare_beneficiary(beneficiary, nblocks):
    if beneficiary is None:
        beneficiary = PED_BENEFICIARY.copy()
    beneficiary = np.asarray(beneficiary)
    if beneficiary.ndim != 1 or beneficiary.shape[0] != nblocks:
        raise ValueError("beneficiary must be a 1D array with one entry (0=self, 1=other) per block")
    ben = beneficiary.astype(int, copy=False)
    if not np.isin(ben, [0, 1]).all():
        raise ValueError("beneficiary entries must be 0 (self) or 1 (other)")
    return ben


# =============================================================================
# ped_par_k -- parabolic prosocial effort discounting, 1 free parameter (k)
#   V_high(E) = r_high - k*E**2   (single k for both self and other blocks)
# =============================================================================
def ped_par_k_sim(params: np.ndarray, payouts=PED_PAYOUTS, effort_levels=PED_EFFORT_LEVELS,
                  beneficiary=PED_BENEFICIARY, seed: int | None = None) -> dict:
    """Simulate the 1-parameter parabolic prosocial effort discounting model (single k)."""
    payouts_bt, effort_levels = _prepare_2opt_inputs(payouts, effort_levels, PED_PAYOUTS, PED_EFFORT_LEVELS, "effort_levels")
    params = np.asarray(params, dtype=float)
    if params.ndim != 2 or params.shape[1] != 1:
        raise ValueError("params must have shape (nsubjects, 1): column [k]")

    nsubjects = params.shape[0]
    nblocks, ntrials = payouts_bt.shape[:2]
    ben = _prepare_beneficiary(beneficiary, nblocks)
    payouts_subj, effort_subj = _expand_2opt_for_subjects(payouts_bt, effort_levels, nsubjects)
    beneficiary_subj = np.broadcast_to(ben[None, :], (nsubjects, nblocks)).copy()
    rng = np.random.default_rng(seed)

    choices = np.empty((nsubjects, nblocks, ntrials), dtype=object)
    choices_high = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    ch_prob = np.zeros((nsubjects, nblocks, ntrials, 2), dtype=float)
    p_high = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    outcomes = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    V_low = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    V_high = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    EV = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    delta_V = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    nll = np.zeros((nsubjects, nblocks, ntrials), dtype=float)

    all_k = params[:, 0]
    if not (all_k > 0.0).all():
        raise ValueError("k must be > 0")

    for s in range(nsubjects):
        k = float(all_k[s])

        for b in range(nblocks):
            E = float(effort_levels[b])

            for t in range(ntrials):
                r_low = payouts_bt[b, t, 0]
                r_high = payouts_bt[b, t, 1]

                V_low_val = r_low
                V_high_val = r_high - k * (E ** 2)

                dv = V_high_val - V_low_val
                p1 = float(expit(dv))
                p = np.array([1.0 - p1, p1])
                c = int(rng.choice([0, 1], p=p))

                ch_prob[s, b, t, :] = p
                p_high[s, b, t] = p1
                delta_V[s, b, t] = dv
                choices[s, b, t] = CHOICE_LABELS_ED[c]
                choices_high[s, b, t] = float(c == 1)
                outcomes[s, b, t] = payouts_bt[b, t, c]
                V_low[s, b, t] = V_low_val
                V_high[s, b, t] = V_high_val
                EV[s, b, t] = V_high_val if c == 1 else V_low_val
                nll[s, b, t] = -np.log(p[c] + 1e-12)

    return {
        "params": params.copy(),
        "payouts": payouts_subj,
        "effort_levels": effort_subj,
        "beneficiary": beneficiary_subj,
        "V_low": V_low,
        "V_high": V_high,
        "EV": EV,
        "delta_V": delta_V,
        "p_high": p_high,
        "choices": choices,
        "ch_prob": ch_prob,
        "choices_high": choices_high,
        "outcomes": outcomes,
        "nll": nll,
        "nll_total": nll.sum(axis=(1, 2)),
    }


def ped_par_k_fit(params, choices, payouts, effort_levels, beneficiary, prior=None, output: str = "npl"):
    """Fit the 1-parameter parabolic prosocial effort discounting model. params: (1,) Gaussian-space [k]."""
    payouts_bt, effort_levels = _prepare_2opt_inputs(payouts, effort_levels, PED_PAYOUTS, PED_EFFORT_LEVELS, "effort_levels")
    nblocks, ntrials = payouts_bt.shape[:2]
    ben = _prepare_beneficiary(beneficiary, nblocks)
    choice_idx = _choices_to_idx(choices, nblocks=nblocks, ntrials=ntrials, choice_map=CHOICE_MAP_ED)

    params = np.asarray(params, dtype=float)
    if params.shape != (1,):
        raise ValueError("params must have shape (1,) in normalized space: [k]")

    k = float(norm2beta(params[0]))
    if not (k > 0.0):
        return 1e7

    V_low = np.zeros((nblocks, ntrials), dtype=float)
    V_high = np.zeros((nblocks, ntrials), dtype=float)
    EV = np.zeros((nblocks, ntrials), dtype=float)
    delta_V = np.zeros((nblocks, ntrials), dtype=float)
    p_high = np.zeros((nblocks, ntrials), dtype=float)
    ch_prob = np.zeros((nblocks, ntrials, 2), dtype=float)
    outcomes = np.zeros((nblocks, ntrials), dtype=float)
    choices_high = np.zeros((nblocks, ntrials), dtype=float)
    nll_trialwise = np.zeros((nblocks, ntrials), dtype=float)
    nll = 0.0

    for b in range(nblocks):
        E = float(effort_levels[b])

        for t in range(ntrials):
            r_low = payouts_bt[b, t, 0]
            r_high = payouts_bt[b, t, 1]

            V_low_val = r_low
            V_high_val = r_high - k * (E ** 2)

            dv = V_high_val - V_low_val
            p1 = float(expit(dv))
            p = np.array([1.0 - p1, p1])
            c = int(choice_idx[b, t])

            delta_V[b, t] = dv
            p_high[b, t] = p1
            ch_prob[b, t, :] = p
            choices_high[b, t] = float(c == 1)
            outcomes[b, t] = payouts_bt[b, t, c]
            V_low[b, t] = V_low_val
            V_high[b, t] = V_high_val
            EV[b, t] = V_high_val if c == 1 else V_low_val
            nll_trialwise[b, t] = -np.log(p[c] + 1e-12)
            nll += nll_trialwise[b, t]

    if output == "all":
        return {
            "params": [k],
            "payouts": payouts_bt,
            "effort_levels": effort_levels.copy(),
            "beneficiary": ben.copy(),
            "V_low": V_low,
            "V_high": V_high,
            "EV": EV,
            "delta_V": delta_V,
            "p_high": p_high,
            "choices": np.asarray(choices, dtype=object).copy(),
            "ch_prob": ch_prob,
            "choices_high": choices_high,
            "outcomes": outcomes,
            "nll": nll,
            "nll_trialwise": nll_trialwise,
        }

    return calc_fval(nll, params, prior=prior, output=output)


ped_par_k_desc = """Parabolic prosocial effort discounting with a single discount rate:
V_high(E) = r_high - k*E**2, applied identically whether the reward benefits
self or other; the low-effort option is undiscounted.
p(high_effort) = sigmoid(V_high - V_low).
Free parameter: k (effort discount rate, >0)."""
ped_par_k_id = "ped_par_k"
ped_par_k_spec = {"prosocial_effort_discounting": {"weight": [], "discount": ["k"]}, "shape": "parabolic", "choice_rule": "sigmoid(delta_V)"}
ped_par_k_model = ModelSpec(
    id=ped_par_k_id, spec=ped_par_k_spec, desc=ped_par_k_desc.strip(),
    params=None, sim=ped_par_k_sim, fit=ped_par_k_fit,
)


# =============================================================================
# ped_par_2k -- parabolic prosocial effort discounting, 2 free params
#   V_high(E) = r_high - k_b*E**2, with k_b = k_self on self blocks and
#   k_other on other blocks. params order: [k_self, k_other].
# =============================================================================
def ped_par_2k_sim(params: np.ndarray, payouts=PED_PAYOUTS, effort_levels=PED_EFFORT_LEVELS,
                   beneficiary=PED_BENEFICIARY, seed: int | None = None) -> dict:
    """Simulate the 2-parameter parabolic prosocial effort discounting model (k_self, k_other)."""
    payouts_bt, effort_levels = _prepare_2opt_inputs(payouts, effort_levels, PED_PAYOUTS, PED_EFFORT_LEVELS, "effort_levels")
    params = np.asarray(params, dtype=float)
    if params.ndim != 2 or params.shape[1] != 2:
        raise ValueError("params must have shape (nsubjects, 2): columns [k_self, k_other]")

    nsubjects = params.shape[0]
    nblocks, ntrials = payouts_bt.shape[:2]
    ben = _prepare_beneficiary(beneficiary, nblocks)
    payouts_subj, effort_subj = _expand_2opt_for_subjects(payouts_bt, effort_levels, nsubjects)
    beneficiary_subj = np.broadcast_to(ben[None, :], (nsubjects, nblocks)).copy()
    rng = np.random.default_rng(seed)

    choices = np.empty((nsubjects, nblocks, ntrials), dtype=object)
    choices_high = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    ch_prob = np.zeros((nsubjects, nblocks, ntrials, 2), dtype=float)
    p_high = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    outcomes = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    V_low = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    V_high = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    EV = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    delta_V = np.zeros((nsubjects, nblocks, ntrials), dtype=float)
    nll = np.zeros((nsubjects, nblocks, ntrials), dtype=float)

    all_k_self = params[:, 0]
    all_k_other = params[:, 1]
    if not (all_k_self > 0.0).all():
        raise ValueError("k_self must be > 0")
    if not (all_k_other > 0.0).all():
        raise ValueError("k_other must be > 0")

    for s in range(nsubjects):
        k_self = float(all_k_self[s])
        k_other = float(all_k_other[s])

        for b in range(nblocks):
            E = float(effort_levels[b])
            k = k_self if ben[b] == 0 else k_other

            for t in range(ntrials):
                r_low = payouts_bt[b, t, 0]
                r_high = payouts_bt[b, t, 1]

                V_low_val = r_low
                V_high_val = r_high - k * (E ** 2)

                dv = V_high_val - V_low_val
                p1 = float(expit(dv))
                p = np.array([1.0 - p1, p1])
                c = int(rng.choice([0, 1], p=p))

                ch_prob[s, b, t, :] = p
                p_high[s, b, t] = p1
                delta_V[s, b, t] = dv
                choices[s, b, t] = CHOICE_LABELS_ED[c]
                choices_high[s, b, t] = float(c == 1)
                outcomes[s, b, t] = payouts_bt[b, t, c]
                V_low[s, b, t] = V_low_val
                V_high[s, b, t] = V_high_val
                EV[s, b, t] = V_high_val if c == 1 else V_low_val
                nll[s, b, t] = -np.log(p[c] + 1e-12)

    return {
        "params": params.copy(),
        "payouts": payouts_subj,
        "effort_levels": effort_subj,
        "beneficiary": beneficiary_subj,
        "V_low": V_low,
        "V_high": V_high,
        "EV": EV,
        "delta_V": delta_V,
        "p_high": p_high,
        "choices": choices,
        "ch_prob": ch_prob,
        "choices_high": choices_high,
        "outcomes": outcomes,
        "nll": nll,
        "nll_total": nll.sum(axis=(1, 2)),
    }


def ped_par_2k_fit(params, choices, payouts, effort_levels, beneficiary, prior=None, output: str = "npl"):
    """Fit the 2-parameter parabolic prosocial effort discounting model. params: (2,) Gaussian-space [k_self, k_other]."""
    payouts_bt, effort_levels = _prepare_2opt_inputs(payouts, effort_levels, PED_PAYOUTS, PED_EFFORT_LEVELS, "effort_levels")
    nblocks, ntrials = payouts_bt.shape[:2]
    ben = _prepare_beneficiary(beneficiary, nblocks)
    choice_idx = _choices_to_idx(choices, nblocks=nblocks, ntrials=ntrials, choice_map=CHOICE_MAP_ED)

    params = np.asarray(params, dtype=float)
    if params.shape != (2,):
        raise ValueError("params must have shape (2,) in normalized space: [k_self, k_other]")

    k_self = float(norm2beta(params[0]))
    k_other = float(norm2beta(params[1]))
    if not (k_self > 0.0):
        return 1e7
    if not (k_other > 0.0):
        return 1e7

    V_low = np.zeros((nblocks, ntrials), dtype=float)
    V_high = np.zeros((nblocks, ntrials), dtype=float)
    EV = np.zeros((nblocks, ntrials), dtype=float)
    delta_V = np.zeros((nblocks, ntrials), dtype=float)
    p_high = np.zeros((nblocks, ntrials), dtype=float)
    ch_prob = np.zeros((nblocks, ntrials, 2), dtype=float)
    outcomes = np.zeros((nblocks, ntrials), dtype=float)
    choices_high = np.zeros((nblocks, ntrials), dtype=float)
    nll_trialwise = np.zeros((nblocks, ntrials), dtype=float)
    nll = 0.0

    for b in range(nblocks):
        E = float(effort_levels[b])
        k = k_self if ben[b] == 0 else k_other

        for t in range(ntrials):
            r_low = payouts_bt[b, t, 0]
            r_high = payouts_bt[b, t, 1]

            V_low_val = r_low
            V_high_val = r_high - k * (E ** 2)

            dv = V_high_val - V_low_val
            p1 = float(expit(dv))
            p = np.array([1.0 - p1, p1])
            c = int(choice_idx[b, t])

            delta_V[b, t] = dv
            p_high[b, t] = p1
            ch_prob[b, t, :] = p
            choices_high[b, t] = float(c == 1)
            outcomes[b, t] = payouts_bt[b, t, c]
            V_low[b, t] = V_low_val
            V_high[b, t] = V_high_val
            EV[b, t] = V_high_val if c == 1 else V_low_val
            nll_trialwise[b, t] = -np.log(p[c] + 1e-12)
            nll += nll_trialwise[b, t]

    if output == "all":
        return {
            "params": [k_self, k_other],
            "payouts": payouts_bt,
            "effort_levels": effort_levels.copy(),
            "beneficiary": ben.copy(),
            "V_low": V_low,
            "V_high": V_high,
            "EV": EV,
            "delta_V": delta_V,
            "p_high": p_high,
            "choices": np.asarray(choices, dtype=object).copy(),
            "ch_prob": ch_prob,
            "choices_high": choices_high,
            "outcomes": outcomes,
            "nll": nll,
            "nll_trialwise": nll_trialwise,
        }

    return calc_fval(nll, params, prior=prior, output=output)


ped_par_2k_desc = """Parabolic prosocial effort discounting with separate self/other rates:
V_high(E) = r_high - k_b*E**2, where k_b = k_self on blocks whose reward
benefits the chooser and k_other on blocks whose reward benefits a social
target; the low-effort option is undiscounted.
p(high_effort) = sigmoid(V_high - V_low).
Free parameters: k_self (effort discount rate for self, >0), k_other (effort
discount rate for other, >0). k_other > k_self indicates steeper effort
discounting when helping others."""
ped_par_2k_id = "ped_par_2k"
ped_par_2k_spec = {"prosocial_effort_discounting": {"weight": [], "discount": ["k_self", "k_other"]}, "shape": "parabolic", "choice_rule": "sigmoid(delta_V)"}
ped_par_2k_model = ModelSpec(
    id=ped_par_2k_id, spec=ped_par_2k_spec, desc=ped_par_2k_desc.strip(),
    params=None, sim=ped_par_2k_sim, fit=ped_par_2k_fit,
)
