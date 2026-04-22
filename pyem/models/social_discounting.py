import numpy as np
from ..utils.math import norm2alpha, norm2beta, calc_fval

DEFAULT_SOCIAL_DISTS = np.array([1, 2, 5, 10, 20, 50, 100], dtype=float)

# Axis convention for payouts:
#   payouts[trial, choice, target]
# where choice 0='selfish', 1='prosocial' and target 0='self', 1='other'.
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
    "selfish": 0,
    "prosocial": 1,
    "self": 0,
    "other": 1,
    "s": 0,
    "p": 1,
    0: 0,
    1: 1,
}


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



def _discount_other_hyperbolic(other_payoff: np.ndarray, k: float, N: float) -> np.ndarray:
    return other_payoff / (1.0 + k * N)



def _discount_other_exponential(other_payoff: np.ndarray, k: float, N: float) -> np.ndarray:
    return other_payoff * np.exp(-k * N)



def _logistic_p_prosocial(delta_v: np.ndarray | float, beta: float = 1.0) -> np.ndarray | float:
    z = np.clip(beta * delta_v, -709.0, 709.0)
    return 1.0 / (1.0 + np.exp(-z))



def _expand_for_subjects(payouts_bt: np.ndarray, social_dists: np.ndarray, nsubjects: int) -> tuple[np.ndarray, np.ndarray]:
    nblocks, ntrials = payouts_bt.shape[:2]
    payouts_subj = np.broadcast_to(payouts_bt[None, ...], (nsubjects, nblocks, ntrials, 2, 2)).copy()
    social_dists_subj = np.broadcast_to(social_dists[None, :], (nsubjects, nblocks)).copy()
    return payouts_subj, social_dists_subj


# -----------------------------------------------------------------------------
# Hyperbolic social discounting with w_self and w_other weights
# Logistic on delta_V = EV_prosocial - EV_selfish
# params sim: (nsubjects, 4) in actual space [beta, w_self, w_other, k]
# params fit: (4,) in normalized space [beta, w_self, w_other, k]
# -----------------------------------------------------------------------------
def hsd_wswo_sim(
    params: np.ndarray,
    payouts: np.ndarray | None = TASK1_PAYOUTS,
    social_dists: np.ndarray | None = DEFAULT_SOCIAL_DISTS,
):
    payouts_bt, social_dists = _prepare_social_inputs(payouts=payouts, social_dists=social_dists)
    params = np.asarray(params, dtype=float)

    if params.ndim != 2 or params.shape[1] != 4:
        raise ValueError("params must have shape (nsubjects, 4)")

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

    all_beta = params[:, 0]
    all_w_self = params[:, 1]
    all_w_other = params[:, 2]
    all_k = params[:, 3]

    if not ((all_beta > 0.0)).all():
        raise ValueError("beta must be > 0")
    if not (((all_w_self >= 0.0) & (all_w_self <= 1.0))).all():
        raise ValueError("w_self must be between 0 and 1")
    if not (((all_w_other >= 0.0) & (all_w_other <= 1.0))).all():
        raise ValueError("w_other must be between 0 and 1")
    if not ((all_k > 0.0)).all():
        raise ValueError("k must be > 0")

    for s in range(nsubjects):
        beta = float(all_beta[s])
        w_self = float(all_w_self[s])
        w_other = float(all_w_other[s])
        k = float(all_k[s])

        for b in range(nblocks):
            N = float(social_dists[b])

            for t in range(ntrials):
                trial_payouts = payouts_bt[b, t, :, :]
                self_pay = trial_payouts[:, 0]
                other_pay = trial_payouts[:, 1]

                U_self_opts = w_self * self_pay
                U_other_opts = w_other * _discount_other_hyperbolic(other_pay, k=k, N=N)
                EV_opts = U_self_opts + U_other_opts

                dv = EV_opts[1] - EV_opts[0]
                p_ps = float(_logistic_p_prosocial(dv, beta=beta))
                p = np.array([1.0 - p_ps, p_ps], dtype=float)
                c = int(rng.choice([0, 1], p=p))

                ch_prob[s, b, t, :] = p
                p_prosocial[s, b, t] = p_ps
                delta_V[s, b, t] = dv
                choices[s, b, t] = CHOICE_LABELS[c]
                choices_prosocial[s, b, t] = float(c == 1)
                outcomes[s, b, t, :] = trial_payouts[c, :]
                U_self[s, b, t] = U_self_opts[c]
                U_other[s, b, t] = U_other_opts[c]
                EV[s, b, t] = EV_opts[c]
                nll[s, b, t] = -np.log(p[c] + 1e-12)

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



def hsd_wswo_fit(
    params: np.ndarray,
    choices: np.ndarray,
    payouts: np.ndarray | None,
    social_dists: np.ndarray,
    prior=None,
    output: str = "npl",
):
    payouts_bt, social_dists = _prepare_social_inputs(payouts=payouts, social_dists=social_dists)
    nblocks, ntrials = payouts_bt.shape[:2]
    choice_idx = _choices_to_idx(choices, nblocks=nblocks, ntrials=ntrials)

    params = np.asarray(params, dtype=float)
    if params.shape != (4,):
        raise ValueError("params must have shape (4,) in normalized space")

    beta = float(norm2beta(params[0]))
    w_self = float(norm2alpha(params[1]))
    w_other = float(norm2alpha(params[2]))
    k = float(norm2beta(params[3]))

    if not (beta > 0.0 and k > 0.0):
        return 1e7
    if not (0.0 <= w_self <= 1.0 and 0.0 <= w_other <= 1.0):
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
            self_pay = trial_payouts[:, 0]
            other_pay = trial_payouts[:, 1]

            U_self_opts = w_self * self_pay
            U_other_opts = w_other * _discount_other_hyperbolic(other_pay, k=k, N=N)
            EV_opts = U_self_opts + U_other_opts

            dv = EV_opts[1] - EV_opts[0]
            p_ps = float(_logistic_p_prosocial(dv, beta=beta))
            p = np.array([1.0 - p_ps, p_ps], dtype=float)
            c = int(choice_idx[b, t])

            delta_V[b, t] = dv
            p_prosocial[b, t] = p_ps
            ch_prob[b, t, :] = p
            choices_prosocial[b, t] = float(c == 1)
            outcomes[b, t, :] = trial_payouts[c, :]
            U_self[b, t] = U_self_opts[c]
            U_other[b, t] = U_other_opts[c]
            EV[b, t] = EV_opts[c]
            nll_trialwise[b, t] = -np.log(p[c] + 1e-12)
            nll += nll_trialwise[b, t]

    if output == "all":
        return {
            "params": [beta, w_self, w_other, k],
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

    return calc_fval(nll, params, prior=prior, output=output)


# -----------------------------------------------------------------------------
# Hyperbolic social discounting without self/other weights, beta free
# Logistic on delta_V = EV_prosocial - EV_selfish
# params sim: (nsubjects, 2) in actual space [beta, k]
# params fit: (2,) in normalized space [beta, k]
# -----------------------------------------------------------------------------
def hsd_softmax_sim(
    params: np.ndarray,
    payouts: np.ndarray | None = TASK1_PAYOUTS,
    social_dists: np.ndarray | None = DEFAULT_SOCIAL_DISTS,
):
    payouts_bt, social_dists = _prepare_social_inputs(payouts=payouts, social_dists=social_dists)
    params = np.asarray(params, dtype=float)

    if params.ndim != 2 or params.shape[1] != 2:
        raise ValueError("params must have shape (nsubjects, 2)")

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

    all_beta = params[:, 0]
    all_k = params[:, 1]

    if not ((all_beta > 0.0)).all():
        raise ValueError("beta must be > 0")
    if not ((all_k > 0.0)).all():
        raise ValueError("k must be > 0")

    for s in range(nsubjects):
        beta = float(all_beta[s])
        k = float(all_k[s])

        for b in range(nblocks):
            N = float(social_dists[b])

            for t in range(ntrials):
                trial_payouts = payouts_bt[b, t, :, :]
                self_pay = trial_payouts[:, 0]
                other_pay = trial_payouts[:, 1]

                U_self_opts = self_pay.copy()
                U_other_opts = _discount_other_hyperbolic(other_pay, k=k, N=N)
                EV_opts = U_self_opts + U_other_opts

                dv = EV_opts[1] - EV_opts[0]
                p_ps = float(_logistic_p_prosocial(dv, beta=beta))
                p = np.array([1.0 - p_ps, p_ps], dtype=float)
                c = int(rng.choice([0, 1], p=p))

                ch_prob[s, b, t, :] = p
                p_prosocial[s, b, t] = p_ps
                delta_V[s, b, t] = dv
                choices[s, b, t] = CHOICE_LABELS[c]
                choices_prosocial[s, b, t] = float(c == 1)
                outcomes[s, b, t, :] = trial_payouts[c, :]
                U_self[s, b, t] = U_self_opts[c]
                U_other[s, b, t] = U_other_opts[c]
                EV[s, b, t] = EV_opts[c]
                nll[s, b, t] = -np.log(p[c] + 1e-12)

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



def hsd_softmax_fit(
    params: np.ndarray,
    choices: np.ndarray,
    payouts: np.ndarray | None,
    social_dists: np.ndarray,
    prior=None,
    output: str = "npl",
):
    payouts_bt, social_dists = _prepare_social_inputs(payouts=payouts, social_dists=social_dists)
    nblocks, ntrials = payouts_bt.shape[:2]
    choice_idx = _choices_to_idx(choices, nblocks=nblocks, ntrials=ntrials)

    params = np.asarray(params, dtype=float)
    if params.shape != (2,):
        raise ValueError("params must have shape (2,) in normalized space")

    beta = float(norm2beta(params[0]))
    k = float(norm2beta(params[1]))

    if not (beta > 0.0 and k > 0.0):
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
            self_pay = trial_payouts[:, 0]
            other_pay = trial_payouts[:, 1]

            U_self_opts = self_pay.copy()
            U_other_opts = _discount_other_hyperbolic(other_pay, k=k, N=N)
            EV_opts = U_self_opts + U_other_opts

            dv = EV_opts[1] - EV_opts[0]
            p_ps = float(_logistic_p_prosocial(dv, beta=beta))
            p = np.array([1.0 - p_ps, p_ps], dtype=float)
            c = int(choice_idx[b, t])

            delta_V[b, t] = dv
            p_prosocial[b, t] = p_ps
            ch_prob[b, t, :] = p
            choices_prosocial[b, t] = float(c == 1)
            outcomes[b, t, :] = trial_payouts[c, :]
            U_self[b, t] = U_self_opts[c]
            U_other[b, t] = U_other_opts[c]
            EV[b, t] = EV_opts[c]
            nll_trialwise[b, t] = -np.log(p[c] + 1e-12)
            nll += nll_trialwise[b, t]

    if output == "all":
        return {
            "params": [beta, k],
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

    return calc_fval(nll, params, prior=prior, output=output)


# -----------------------------------------------------------------------------
# Hyperbolic social discounting without self/other weights, beta fixed at 1
# Logistic on delta_V = EV_prosocial - EV_selfish
# params sim: (nsubjects, 1) in actual space [k]
# params fit: (1,) in normalized space [k]
# -----------------------------------------------------------------------------
def hsd_sim(
    params: np.ndarray,
    payouts: np.ndarray | None = TASK1_PAYOUTS,
    social_dists: np.ndarray | None = DEFAULT_SOCIAL_DISTS,
):
    payouts_bt, social_dists = _prepare_social_inputs(payouts=payouts, social_dists=social_dists)
    params = np.asarray(params, dtype=float)

    if params.ndim != 2 or params.shape[1] != 1:
        raise ValueError("params must have shape (nsubjects, 1)")

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
    if not ((all_k > 0.0)).all():
        raise ValueError("k must be > 0")

    for s in range(nsubjects):
        k = float(all_k[s])

        for b in range(nblocks):
            N = float(social_dists[b])

            for t in range(ntrials):
                trial_payouts = payouts_bt[b, t, :, :]
                self_pay = trial_payouts[:, 0]
                other_pay = trial_payouts[:, 1]

                U_self_opts = self_pay.copy()
                U_other_opts = _discount_other_hyperbolic(other_pay, k=k, N=N)
                EV_opts = U_self_opts + U_other_opts

                dv = EV_opts[1] - EV_opts[0]
                p_ps = float(_logistic_p_prosocial(dv, beta=1.0))
                p = np.array([1.0 - p_ps, p_ps], dtype=float)
                c = int(rng.choice([0, 1], p=p))

                ch_prob[s, b, t, :] = p
                p_prosocial[s, b, t] = p_ps
                delta_V[s, b, t] = dv
                choices[s, b, t] = CHOICE_LABELS[c]
                choices_prosocial[s, b, t] = float(c == 1)
                outcomes[s, b, t, :] = trial_payouts[c, :]
                U_self[s, b, t] = U_self_opts[c]
                U_other[s, b, t] = U_other_opts[c]
                EV[s, b, t] = EV_opts[c]
                nll[s, b, t] = -np.log(p[c] + 1e-12)

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



def hsd_fit(
    params: np.ndarray,
    choices: np.ndarray,
    payouts: np.ndarray | None,
    social_dists: np.ndarray,
    prior=None,
    output: str = "npl",
):
    payouts_bt, social_dists = _prepare_social_inputs(payouts=payouts, social_dists=social_dists)
    nblocks, ntrials = payouts_bt.shape[:2]
    choice_idx = _choices_to_idx(choices, nblocks=nblocks, ntrials=ntrials)

    params = np.asarray(params, dtype=float)
    if params.shape != (1,):
        raise ValueError("params must have shape (1,) in normalized space")

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
            self_pay = trial_payouts[:, 0]
            other_pay = trial_payouts[:, 1]

            U_self_opts = self_pay.copy()
            U_other_opts = _discount_other_hyperbolic(other_pay, k=k, N=N)
            EV_opts = U_self_opts + U_other_opts

            dv = EV_opts[1] - EV_opts[0]
            p_ps = float(_logistic_p_prosocial(dv, beta=1.0))
            p = np.array([1.0 - p_ps, p_ps], dtype=float)
            c = int(choice_idx[b, t])

            delta_V[b, t] = dv
            p_prosocial[b, t] = p_ps
            ch_prob[b, t, :] = p
            choices_prosocial[b, t] = float(c == 1)
            outcomes[b, t, :] = trial_payouts[c, :]
            U_self[b, t] = U_self_opts[c]
            U_other[b, t] = U_other_opts[c]
            EV[b, t] = EV_opts[c]
            nll_trialwise[b, t] = -np.log(p[c] + 1e-12)
            nll += nll_trialwise[b, t]

    if output == "all":
        return {
            "params": [k],
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

    return calc_fval(nll, params, prior=prior, output=output)


# -----------------------------------------------------------------------------
# Exponential social discounting with w_self and w_other weights
# Logistic on delta_V = EV_prosocial - EV_selfish
# params sim: (nsubjects, 4) in actual space [beta, w_self, w_other, k]
# params fit: (4,) in normalized space [beta, w_self, w_other, k]
# -----------------------------------------------------------------------------
def esd_wswo_sim(
    params: np.ndarray,
    payouts: np.ndarray | None = TASK1_PAYOUTS,
    social_dists: np.ndarray | None = DEFAULT_SOCIAL_DISTS,
):
    payouts_bt, social_dists = _prepare_social_inputs(payouts=payouts, social_dists=social_dists)
    params = np.asarray(params, dtype=float)

    if params.ndim != 2 or params.shape[1] != 4:
        raise ValueError("params must have shape (nsubjects, 4)")

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

    all_beta = params[:, 0]
    all_w_self = params[:, 1]
    all_w_other = params[:, 2]
    all_k = params[:, 3]

    if not ((all_beta > 0.0)).all():
        raise ValueError("beta must be > 0")
    if not (((all_w_self >= 0.0) & (all_w_self <= 1.0))).all():
        raise ValueError("w_self must be between 0 and 1")
    if not (((all_w_other >= 0.0) & (all_w_other <= 1.0))).all():
        raise ValueError("w_other must be between 0 and 1")
    if not ((all_k > 0.0)).all():
        raise ValueError("k must be > 0")

    for s in range(nsubjects):
        beta = float(all_beta[s])
        w_self = float(all_w_self[s])
        w_other = float(all_w_other[s])
        k = float(all_k[s])

        for b in range(nblocks):
            N = float(social_dists[b])

            for t in range(ntrials):
                trial_payouts = payouts_bt[b, t, :, :]
                self_pay = trial_payouts[:, 0]
                other_pay = trial_payouts[:, 1]

                U_self_opts = w_self * self_pay
                U_other_opts = w_other * _discount_other_exponential(other_pay, k=k, N=N)
                EV_opts = U_self_opts + U_other_opts

                dv = EV_opts[1] - EV_opts[0]
                p_ps = float(_logistic_p_prosocial(dv, beta=beta))
                p = np.array([1.0 - p_ps, p_ps], dtype=float)
                c = int(rng.choice([0, 1], p=p))

                ch_prob[s, b, t, :] = p
                p_prosocial[s, b, t] = p_ps
                delta_V[s, b, t] = dv
                choices[s, b, t] = CHOICE_LABELS[c]
                choices_prosocial[s, b, t] = float(c == 1)
                outcomes[s, b, t, :] = trial_payouts[c, :]
                U_self[s, b, t] = U_self_opts[c]
                U_other[s, b, t] = U_other_opts[c]
                EV[s, b, t] = EV_opts[c]
                nll[s, b, t] = -np.log(p[c] + 1e-12)

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



def esd_wswo_fit(
    params: np.ndarray,
    choices: np.ndarray,
    payouts: np.ndarray | None,
    social_dists: np.ndarray,
    prior=None,
    output: str = "npl",
):
    payouts_bt, social_dists = _prepare_social_inputs(payouts=payouts, social_dists=social_dists)
    nblocks, ntrials = payouts_bt.shape[:2]
    choice_idx = _choices_to_idx(choices, nblocks=nblocks, ntrials=ntrials)

    params = np.asarray(params, dtype=float)
    if params.shape != (4,):
        raise ValueError("params must have shape (4,) in normalized space")

    beta = float(norm2beta(params[0]))
    w_self = float(norm2alpha(params[1]))
    w_other = float(norm2alpha(params[2]))
    k = float(norm2beta(params[3]))

    if not (beta > 0.0 and k > 0.0):
        return 1e7
    if not (0.0 <= w_self <= 1.0 and 0.0 <= w_other <= 1.0):
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
            self_pay = trial_payouts[:, 0]
            other_pay = trial_payouts[:, 1]

            U_self_opts = w_self * self_pay
            U_other_opts = w_other * _discount_other_exponential(other_pay, k=k, N=N)
            EV_opts = U_self_opts + U_other_opts

            dv = EV_opts[1] - EV_opts[0]
            p_ps = float(_logistic_p_prosocial(dv, beta=beta))
            p = np.array([1.0 - p_ps, p_ps], dtype=float)
            c = int(choice_idx[b, t])

            delta_V[b, t] = dv
            p_prosocial[b, t] = p_ps
            ch_prob[b, t, :] = p
            choices_prosocial[b, t] = float(c == 1)
            outcomes[b, t, :] = trial_payouts[c, :]
            U_self[b, t] = U_self_opts[c]
            U_other[b, t] = U_other_opts[c]
            EV[b, t] = EV_opts[c]
            nll_trialwise[b, t] = -np.log(p[c] + 1e-12)
            nll += nll_trialwise[b, t]

    if output == "all":
        return {
            "params": [beta, w_self, w_other, k],
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

    return calc_fval(nll, params, prior=prior, output=output)


# -----------------------------------------------------------------------------
# Exponential social discounting without self/other weights
# Logistic on delta_V = EV_prosocial - EV_selfish
# params sim: (nsubjects, 2) in actual space [beta, k]
# params fit: (2,) in normalized space [beta, k]
# -----------------------------------------------------------------------------
def esd_sim(
    params: np.ndarray,
    payouts: np.ndarray | None = TASK1_PAYOUTS,
    social_dists: np.ndarray | None = DEFAULT_SOCIAL_DISTS,
):
    payouts_bt, social_dists = _prepare_social_inputs(payouts=payouts, social_dists=social_dists)
    params = np.asarray(params, dtype=float)

    if params.ndim != 2 or params.shape[1] != 2:
        raise ValueError("params must have shape (nsubjects, 2)")

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

    all_beta = params[:, 0]
    all_k = params[:, 1]

    if not ((all_beta > 0.0)).all():
        raise ValueError("beta must be > 0")
    if not ((all_k > 0.0)).all():
        raise ValueError("k must be > 0")

    for s in range(nsubjects):
        beta = float(all_beta[s])
        k = float(all_k[s])

        for b in range(nblocks):
            N = float(social_dists[b])

            for t in range(ntrials):
                trial_payouts = payouts_bt[b, t, :, :]
                self_pay = trial_payouts[:, 0]
                other_pay = trial_payouts[:, 1]

                U_self_opts = self_pay.copy()
                U_other_opts = _discount_other_exponential(other_pay, k=k, N=N)
                EV_opts = U_self_opts + U_other_opts

                dv = EV_opts[1] - EV_opts[0]
                p_ps = float(_logistic_p_prosocial(dv, beta=beta))
                p = np.array([1.0 - p_ps, p_ps], dtype=float)
                c = int(rng.choice([0, 1], p=p))

                ch_prob[s, b, t, :] = p
                p_prosocial[s, b, t] = p_ps
                delta_V[s, b, t] = dv
                choices[s, b, t] = CHOICE_LABELS[c]
                choices_prosocial[s, b, t] = float(c == 1)
                outcomes[s, b, t, :] = trial_payouts[c, :]
                U_self[s, b, t] = U_self_opts[c]
                U_other[s, b, t] = U_other_opts[c]
                EV[s, b, t] = EV_opts[c]
                nll[s, b, t] = -np.log(p[c] + 1e-12)

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



def esd_fit(
    params: np.ndarray,
    choices: np.ndarray,
    payouts: np.ndarray | None,
    social_dists: np.ndarray,
    prior=None,
    output: str = "npl",
):
    payouts_bt, social_dists = _prepare_social_inputs(payouts=payouts, social_dists=social_dists)
    nblocks, ntrials = payouts_bt.shape[:2]
    choice_idx = _choices_to_idx(choices, nblocks=nblocks, ntrials=ntrials)

    params = np.asarray(params, dtype=float)
    if params.shape != (2,):
        raise ValueError("params must have shape (2,) in normalized space")

    beta = float(norm2beta(params[0]))
    k = float(norm2beta(params[1]))

    if not (beta > 0.0 and k > 0.0):
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
            self_pay = trial_payouts[:, 0]
            other_pay = trial_payouts[:, 1]

            U_self_opts = self_pay.copy()
            U_other_opts = _discount_other_exponential(other_pay, k=k, N=N)
            EV_opts = U_self_opts + U_other_opts

            dv = EV_opts[1] - EV_opts[0]
            p_ps = float(_logistic_p_prosocial(dv, beta=beta))
            p = np.array([1.0 - p_ps, p_ps], dtype=float)
            c = int(choice_idx[b, t])

            delta_V[b, t] = dv
            p_prosocial[b, t] = p_ps
            ch_prob[b, t, :] = p
            choices_prosocial[b, t] = float(c == 1)
            outcomes[b, t, :] = trial_payouts[c, :]
            U_self[b, t] = U_self_opts[c]
            U_other[b, t] = U_other_opts[c]
            EV[b, t] = EV_opts[c]
            nll_trialwise[b, t] = -np.log(p[c] + 1e-12)
            nll += nll_trialwise[b, t]

    if output == "all":
        return {
            "params": [beta, k],
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

    return calc_fval(nll, params, prior=prior, output=output)
