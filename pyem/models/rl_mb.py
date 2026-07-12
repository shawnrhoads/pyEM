"""
Model-based / model-free reinforcement learning for the Daw et al. (2011)
two-step task.

This family file implements the three learners described in the Supplemental
Experimental Procedures of Daw, Gershman, Seymour, Dayan & Dolan (2011),
"Model-based influences on humans' choices and striatal prediction errors"
(Neuron 69, 1204-1215), and the joint likelihood coded in the authors'
``llm2b2alr.m``:

* ``sarsa_lambda``  - pure model-free SARSA(lambda) learner (w = 0)
* ``model_based``   - pure model-based Bellman learner (w = 1)
* ``hybrid_mbmf``   - the hybrid that mixes the two with weight w (= omega)

All three are nested versions of the same trial-by-trial equations, so the
per-trial update/choice logic lives once in the private ``_twostep_*`` helpers
and each public ``*_sim`` / ``*_fit`` pair is a thin wrapper that unpacks its
own parameter subset, applies the Gaussian->natural transforms, and bounds-
checks.  This mirrors the multi-model layout of ``pyem/models/rl.py``.

Parameters are fit in **Gaussian (unbounded)** space and transformed to their
natural ranges inside ``*_fit`` (and supplied directly in natural space to
``*_sim``).  Following ``llm2b2alr.m`` the transforms are::

    beta1, beta2            exp(x)              (0, inf)    softmax inverse temps
    alpha1, alpha2, lambda  1/(1+exp(-x))       (0, 1)      learning rates / trace
    omega (= w)             1/(1+exp(-x))       (0, 1)      model-based weight
    r     (= p)             x                   (-inf, inf) first-stage stickiness

Note two deliberate deviations from generic pyEM helpers / conventions, both
required to match the source material:

* ``beta1``/``beta2`` use a plain ``exp(x)`` transform rather than
  :func:`pyem.utils.math.norm2beta` (which saturates at 20); Daw's model puts
  no finite ceiling on the inverse temperatures.
* The model-based back-up uses the hard ``max`` over second-stage actions, as
  written in the supplement's Bellman equation.  ``llm2b2alr.m`` instead uses a
  softmax with a fixed inverse temperature of 20 as a smooth surrogate for that
  max; the two agree closely but the supplement equation is authoritative here.
"""

import numpy as np
from typing import Dict

from ..utils.math import softmax, norm2alpha, calc_fval
from ..core.modelspec import ModelSpec

# ---------------------------------------------------------------------------
# Gaussian -> natural transforms that pyem.utils.math does not provide.
# norm2alpha (logistic) covers the (0, 1) parameters; these two cover the
# positive-unbounded betas and the real-valued stickiness term.
# ---------------------------------------------------------------------------

def _exp(x):
    """Gaussian -> (0, inf); matches ``b = exp(x)`` in llm2b2alr.m."""
    return np.exp(x)

def _identity(x):
    """Gaussian -> (-inf, inf); matches ``rep = x*eye(2)`` in llm2b2alr.m."""
    return x

# Task constants (Daw et al., 2011). The common first-stage transition occurs
# with probability 0.7 and maps action a -> second-stage state a; the rare
# transition (0.3) inverts it. The estimated transition matrix snaps between
# these two hypotheses depending on the experienced counts (supplement p.2).
_P_COMMON = 0.70
_TR_COMMON = np.array([[0.7, 0.3], [0.3, 0.7]])   # Tr[s', a] when a<->s' is common
_TR_RARE = np.array([[0.3, 0.7], [0.7, 0.3]])
# Second-stage reward-probability random walk (reflecting bounds / diffusion).
_RW_LO, _RW_HI, _RW_SD = 0.25, 0.75, 0.025


def _estimate_transition(n: np.ndarray) -> np.ndarray:
    """Count-based transition estimate.

    ``n[s2, a1]`` counts transitions to second-stage state ``s2`` after first-
    stage action ``a1``.  If the diagonal (a0->s0, a1->s1) dominates, action a
    commonly leads to state a; otherwise the mapping is inverted.  Returns
    ``Tr[s', a] = P_hat(s' | a)``.
    """
    if n[0, 0] + n[1, 1] > n[0, 1] + n[1, 0]:
        return _TR_COMMON
    return _TR_RARE


def _gen_reward_walks(ntrials: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian random walks for the four second-stage payoff probabilities.

    Returns an array of shape ``(2 states, 2 actions, ntrials)`` with values in
    ``[0.25, 0.75]`` (reflecting boundaries), matching the empirical range of
    ``DawEtAl11RandomWalks.mat``.
    """
    walks = np.zeros((2, 2, ntrials), dtype=float)
    p = rng.uniform(_RW_LO, _RW_HI, size=(2, 2))
    for t in range(ntrials):
        walks[:, :, t] = p
        p = p + _RW_SD * rng.standard_normal((2, 2))
        # reflect at the boundaries, then clip for numerical safety
        p = np.where(p > _RW_HI, 2 * _RW_HI - p, p)
        p = np.where(p < _RW_LO, 2 * _RW_LO - p, p)
        p = np.clip(p, _RW_LO, _RW_HI)
    return walks


# ---------------------------------------------------------------------------
# Shared trial-by-trial core (used by every model; nesting is via the natural-
# space params passed in: omega=0 -> pure MF, omega=1 -> pure MB).
# ---------------------------------------------------------------------------

def _twostep_negll(beta1, beta2, alpha1, alpha2, lam, omega, rep,
                   choices1, states2, choices2, rewards):
    """Negative log-likelihood of one subject's two-step data.

    All parameters are in natural space.  ``choices1``/``states2``/``choices2``
    are integer arrays in ``{0, 1}`` of length ``ntrials``; ``rewards`` is a
    ``{0, 1}`` array of the same length.
    """
    ntrials = len(choices1)
    Q1 = np.zeros(2)              # model-free first-stage values
    Q2 = np.zeros((2, 2))         # second-stage values, Q2[action, state]
    n = np.zeros((2, 2))          # transition counts, n[state, action]
    nll = 0.0
    a1_prev = -1

    for t in range(ntrials):
        # --- model-based first-stage values (Bellman, hard max) ---
        Tr = _estimate_transition(n)          # Tr[s', a] = P_hat(s' | a)
        V2 = Q2.max(axis=0)                    # max over actions -> value per state
        Qmb = Tr.T @ V2                        # Qmb[a] = sum_s' P(s'|a) * V2(s')

        # --- net first-stage value + perseveration ---
        Qnet = omega * Qmb + (1.0 - omega) * Q1
        if a1_prev >= 0:
            Qnet = Qnet.copy()
            Qnet[a1_prev] += rep

        a1 = int(choices1[t])
        p1 = softmax(Qnet, beta1)
        nll += -np.log(p1[a1] + 1e-12)

        s2 = int(states2[t])
        a2 = int(choices2[t])
        p2 = softmax(Q2[:, s2], beta2)
        nll += -np.log(p2[a2] + 1e-12)

        r = float(rewards[t])
        # SARSA(lambda) prediction errors
        de1 = Q2[a2, s2] - Q1[a1]
        de2 = r - Q2[a2, s2]
        Q1[a1] += alpha1 * (de1 + lam * de2)
        Q2[a2, s2] += alpha2 * de2

        n[s2, a1] += 1
        a1_prev = a1

    return nll


def _twostep_sim(beta1, beta2, alpha1, alpha2, lam, omega, rep, ntrials, rng):
    """Simulate one subject; returns integer choice/state arrays and rewards."""
    Q1 = np.zeros(2)
    Q2 = np.zeros((2, 2))
    n = np.zeros((2, 2))
    a1_prev = -1

    walks = _gen_reward_walks(ntrials, rng)   # (state, action, trial)
    choices1 = np.zeros(ntrials, dtype=int)
    states2 = np.zeros(ntrials, dtype=int)
    choices2 = np.zeros(ntrials, dtype=int)
    rewards = np.zeros(ntrials, dtype=float)
    ch_prob1 = np.zeros((ntrials, 2), dtype=float)
    ch_prob2 = np.zeros((ntrials, 2), dtype=float)

    for t in range(ntrials):
        Tr = _estimate_transition(n)
        V2 = Q2.max(axis=0)
        Qmb = Tr.T @ V2
        Qnet = omega * Qmb + (1.0 - omega) * Q1
        if a1_prev >= 0:
            Qnet = Qnet.copy()
            Qnet[a1_prev] += rep

        p1 = softmax(Qnet, beta1)
        ch_prob1[t, :] = p1
        a1 = int(rng.choice(2, p=p1))
        choices1[t] = a1

        # common (0.7) transition maps action a -> state a; rare inverts it
        s2 = a1 if (rng.random() < _P_COMMON) else 1 - a1
        states2[t] = s2

        p2 = softmax(Q2[:, s2], beta2)
        ch_prob2[t, :] = p2
        a2 = int(rng.choice(2, p=p2))
        choices2[t] = a2

        r = 1.0 if (walks[s2, a2, t] > rng.random()) else 0.0
        rewards[t] = r

        de1 = Q2[a2, s2] - Q1[a1]
        de2 = r - Q2[a2, s2]
        Q1[a1] += alpha1 * (de1 + lam * de2)
        Q2[a2, s2] += alpha2 * de2

        n[s2, a1] += 1
        a1_prev = a1

    return {
        "choices1": choices1,
        "states2": states2,
        "choices2": choices2,
        "rewards": rewards,
        "ch_prob1": ch_prob1,
        "ch_prob2": ch_prob2,
        "rewprob": walks,
    }


def _alloc_sim(nsubjects: int, ntrials: int) -> Dict[str, np.ndarray]:
    return dict(
        choices1=np.zeros((nsubjects, ntrials), dtype=int),
        states2=np.zeros((nsubjects, ntrials), dtype=int),
        choices2=np.zeros((nsubjects, ntrials), dtype=int),
        rewards=np.zeros((nsubjects, ntrials), dtype=float),
        ch_prob1=np.zeros((nsubjects, ntrials, 2), dtype=float),
        ch_prob2=np.zeros((nsubjects, ntrials, 2), dtype=float),
        rewprob=np.zeros((nsubjects, 2, 2, ntrials), dtype=float),
    )


# ===========================================================================
# 1. Model-free SARSA(lambda)  (w = 0)
#    Free parameters: beta1, beta2, alpha1, alpha2, lambda, r
# ===========================================================================

def sarsa_lambda_sim(params: np.ndarray, ntrials: int = 200, seed: int | None = None):
    """Simulate the pure model-free SARSA(lambda) learner (Daw et al., 2011).

    ``params`` are **natural**-space, shape ``(nsubjects, 6)`` =
    ``[beta1, beta2, alpha1, alpha2, lambda, r]``.
    """
    if params.ndim != 2 or params.shape[1] != 6:
        raise ValueError("params must be (nsubjects, 6) = "
                         "[beta1, beta2, alpha1, alpha2, lambda, r]")
    nsubjects = params.shape[0]
    b1, b2, a1, a2, lam, rep = (params[:, i].astype(float) for i in range(6))

    if not ((b1 > 0) & np.isfinite(b1)).all():
        raise ValueError("beta1 out of bounds (0, inf)")
    if not ((b2 > 0) & np.isfinite(b2)).all():
        raise ValueError("beta2 out of bounds (0, inf)")
    for arr, name in [(a1, "alpha1"), (a2, "alpha2"), (lam, "lambda")]:
        if not ((arr >= 0.0) & (arr <= 1.0)).all():
            raise ValueError(f"{name} out of bounds [0, 1]")
    if not np.isfinite(rep).all():
        raise ValueError("r must be finite")

    rng = np.random.default_rng(seed)
    dat = _alloc_sim(nsubjects, ntrials)
    for s in range(nsubjects):
        out = _twostep_sim(b1[s], b2[s], a1[s], a2[s], lam[s], 0.0, rep[s],
                           ntrials, rng)
        for k in ("choices1", "states2", "choices2", "rewards",
                  "ch_prob1", "ch_prob2"):
            dat[k][s] = out[k]
        dat["rewprob"][s] = out["rewprob"]

    return {
        "params": np.column_stack([b1, b2, a1, a2, lam, rep]),
        "choices1": dat["choices1"],
        "states2": dat["states2"],
        "choices2": dat["choices2"],
        "rewards": dat["rewards"],
        "ch_prob1": dat["ch_prob1"],
        "ch_prob2": dat["ch_prob2"],
        "rewprob": dat["rewprob"],
    }


def sarsa_lambda_fit(params, choices1, states2, choices2, rewards,
                     prior=None, output="npl"):
    """EM-compatible NPL/NLL for the model-free SARSA(lambda) learner.

    ``params``: (6,) in Gaussian space = [beta1, beta2, alpha1, alpha2, lambda, r].
    """
    beta1 = _exp(params[0])
    beta2 = _exp(params[1])
    alpha1 = norm2alpha(params[2])
    alpha2 = norm2alpha(params[3])
    lam = norm2alpha(params[4])
    rep = _identity(params[5])

    # one bounds check per free parameter
    if not (np.isfinite(beta1) and beta1 > 0):
        return 1e7
    if not (np.isfinite(beta2) and beta2 > 0):
        return 1e7
    if not (0.0 <= alpha1 <= 1.0):
        return 1e7
    if not (0.0 <= alpha2 <= 1.0):
        return 1e7
    if not (0.0 <= lam <= 1.0):
        return 1e7
    if not np.isfinite(rep):
        return 1e7

    nll = _twostep_negll(beta1, beta2, alpha1, alpha2, lam, 0.0, rep,
                         choices1, states2, choices2, rewards)

    if output == "all":
        return {
            "params": [beta1, beta2, alpha1, alpha2, lam, rep],
            "choices1": choices1,
            "states2": states2,
            "choices2": choices2,
            "rewards": rewards,
            "nll": nll,
        }
    return calc_fval(nll, params, prior=prior, output=output)


sarsa_lambda_desc = """Model-free SARSA(lambda) temporal-difference learner for
the Daw et al. (2011) two-step task (w = 0 special case of the hybrid). The
first-stage value is updated toward the experienced second-stage value plus a
lambda-weighted stage-skipping update by the reward prediction error; choices
at each stage are softmax over the learned values, with first-stage
perseveration. Free parameters: beta1, beta2, alpha1, alpha2, lambda, r."""
sarsa_lambda_id = "sarsa_lambda"
sarsa_lambda_spec = {
    "rl": {
        "softmax": ["beta1", "beta2"],
        "sarsa": ["alpha1", "alpha2", "lambda"],
        "choice": ["r"],
    }
}
sarsa_lambda_model = ModelSpec(
    id=sarsa_lambda_id, spec=sarsa_lambda_spec, desc=sarsa_lambda_desc.strip(),
    params=None, sim=sarsa_lambda_sim, fit=sarsa_lambda_fit,
)


# ===========================================================================
# 2. Model-based Bellman learner  (w = 1; alpha1 and lambda drop out)
#    Free parameters: beta1, beta2, alpha2, r
# ===========================================================================

def model_based_sim(params: np.ndarray, ntrials: int = 200, seed: int | None = None):
    """Simulate the pure model-based Bellman learner (Daw et al., 2011).

    With w = 1 the model-free first-stage values never enter the choice, so
    alpha1 and lambda are unidentifiable and are not free parameters.

    ``params`` are **natural**-space, shape ``(nsubjects, 4)`` =
    ``[beta1, beta2, alpha2, r]``.
    """
    if params.ndim != 2 or params.shape[1] != 4:
        raise ValueError("params must be (nsubjects, 4) = "
                         "[beta1, beta2, alpha2, r]")
    nsubjects = params.shape[0]
    b1, b2, a2, rep = (params[:, i].astype(float) for i in range(4))

    if not ((b1 > 0) & np.isfinite(b1)).all():
        raise ValueError("beta1 out of bounds (0, inf)")
    if not ((b2 > 0) & np.isfinite(b2)).all():
        raise ValueError("beta2 out of bounds (0, inf)")
    if not ((a2 >= 0.0) & (a2 <= 1.0)).all():
        raise ValueError("alpha2 out of bounds [0, 1]")
    if not np.isfinite(rep).all():
        raise ValueError("r must be finite")

    rng = np.random.default_rng(seed)
    dat = _alloc_sim(nsubjects, ntrials)
    for s in range(nsubjects):
        # alpha1 and lambda are irrelevant when omega=1; pass 0.0
        out = _twostep_sim(b1[s], b2[s], 0.0, a2[s], 0.0, 1.0, rep[s],
                           ntrials, rng)
        for k in ("choices1", "states2", "choices2", "rewards",
                  "ch_prob1", "ch_prob2"):
            dat[k][s] = out[k]
        dat["rewprob"][s] = out["rewprob"]

    return {
        "params": np.column_stack([b1, b2, a2, rep]),
        "choices1": dat["choices1"],
        "states2": dat["states2"],
        "choices2": dat["choices2"],
        "rewards": dat["rewards"],
        "ch_prob1": dat["ch_prob1"],
        "ch_prob2": dat["ch_prob2"],
        "rewprob": dat["rewprob"],
    }


def model_based_fit(params, choices1, states2, choices2, rewards,
                    prior=None, output="npl"):
    """EM-compatible NPL/NLL for the model-based Bellman learner.

    ``params``: (4,) in Gaussian space = [beta1, beta2, alpha2, r].
    """
    beta1 = _exp(params[0])
    beta2 = _exp(params[1])
    alpha2 = norm2alpha(params[2])
    rep = _identity(params[3])

    # one bounds check per free parameter
    if not (np.isfinite(beta1) and beta1 > 0):
        return 1e7
    if not (np.isfinite(beta2) and beta2 > 0):
        return 1e7
    if not (0.0 <= alpha2 <= 1.0):
        return 1e7
    if not np.isfinite(rep):
        return 1e7

    nll = _twostep_negll(beta1, beta2, 0.0, alpha2, 0.0, 1.0, rep,
                         choices1, states2, choices2, rewards)

    if output == "all":
        return {
            "params": [beta1, beta2, alpha2, rep],
            "choices1": choices1,
            "states2": states2,
            "choices2": choices2,
            "rewards": rewards,
            "nll": nll,
        }
    return calc_fval(nll, params, prior=prior, output=output)


model_based_desc = """Model-based Bellman learner for the Daw et al. (2011)
two-step task (w = 1 special case of the hybrid). First-stage values are
recomputed each trial from the (count-estimated) transition structure and the
learned second-stage values via Bellman's equation; choices are softmax with
first-stage perseveration. Because the model-free first-stage values never
enter the choice, alpha1 and lambda drop out. Free parameters: beta1, beta2,
alpha2, r."""
model_based_id = "model_based"
model_based_spec = {
    "rl": {
        "softmax": ["beta1", "beta2"],
        "bellman": ["alpha2"],
        "choice": ["r"],
    }
}
model_based_model = ModelSpec(
    id=model_based_id, spec=model_based_spec, desc=model_based_desc.strip(),
    params=None, sim=model_based_sim, fit=model_based_fit,
)


# ===========================================================================
# 3. Hybrid model-based / model-free  (w = omega free; 7 parameters)
#    Free parameters: beta1, beta2, alpha1, alpha2, lambda, omega, r
# ===========================================================================

def hybrid_mbmf_sim(params: np.ndarray, ntrials: int = 200, seed: int | None = None):
    """Simulate the hybrid MB/MF learner (Daw et al., 2011; llm2b2alr.m).

    ``params`` are **natural**-space, shape ``(nsubjects, 7)`` =
    ``[beta1, beta2, alpha1, alpha2, lambda, omega, r]``.
    """
    if params.ndim != 2 or params.shape[1] != 7:
        raise ValueError("params must be (nsubjects, 7) = "
                         "[beta1, beta2, alpha1, alpha2, lambda, omega, r]")
    nsubjects = params.shape[0]
    b1, b2, a1, a2, lam, om, rep = (params[:, i].astype(float) for i in range(7))

    if not ((b1 > 0) & np.isfinite(b1)).all():
        raise ValueError("beta1 out of bounds (0, inf)")
    if not ((b2 > 0) & np.isfinite(b2)).all():
        raise ValueError("beta2 out of bounds (0, inf)")
    for arr, name in [(a1, "alpha1"), (a2, "alpha2"), (lam, "lambda"),
                      (om, "omega")]:
        if not ((arr >= 0.0) & (arr <= 1.0)).all():
            raise ValueError(f"{name} out of bounds [0, 1]")
    if not np.isfinite(rep).all():
        raise ValueError("r must be finite")

    rng = np.random.default_rng(seed)
    dat = _alloc_sim(nsubjects, ntrials)
    for s in range(nsubjects):
        out = _twostep_sim(b1[s], b2[s], a1[s], a2[s], lam[s], om[s], rep[s],
                           ntrials, rng)
        for k in ("choices1", "states2", "choices2", "rewards",
                  "ch_prob1", "ch_prob2"):
            dat[k][s] = out[k]
        dat["rewprob"][s] = out["rewprob"]

    return {
        "params": np.column_stack([b1, b2, a1, a2, lam, om, rep]),
        "choices1": dat["choices1"],
        "states2": dat["states2"],
        "choices2": dat["choices2"],
        "rewards": dat["rewards"],
        "ch_prob1": dat["ch_prob1"],
        "ch_prob2": dat["ch_prob2"],
        "rewprob": dat["rewprob"],
    }


def hybrid_mbmf_fit(params, choices1, states2, choices2, rewards,
                    prior=None, output="npl"):
    """EM-compatible NPL/NLL for the hybrid MB/MF learner.

    ``params``: (7,) in Gaussian space =
    [beta1, beta2, alpha1, alpha2, lambda, omega, r].
    """
    beta1 = _exp(params[0])
    beta2 = _exp(params[1])
    alpha1 = norm2alpha(params[2])
    alpha2 = norm2alpha(params[3])
    lam = norm2alpha(params[4])
    omega = norm2alpha(params[5])
    rep = _identity(params[6])

    # one bounds check per free parameter
    if not (np.isfinite(beta1) and beta1 > 0):
        return 1e7
    if not (np.isfinite(beta2) and beta2 > 0):
        return 1e7
    if not (0.0 <= alpha1 <= 1.0):
        return 1e7
    if not (0.0 <= alpha2 <= 1.0):
        return 1e7
    if not (0.0 <= lam <= 1.0):
        return 1e7
    if not (0.0 <= omega <= 1.0):
        return 1e7
    if not np.isfinite(rep):
        return 1e7

    nll = _twostep_negll(beta1, beta2, alpha1, alpha2, lam, omega, rep,
                         choices1, states2, choices2, rewards)

    if output == "all":
        return {
            "params": [beta1, beta2, alpha1, alpha2, lam, omega, rep],
            "choices1": choices1,
            "states2": states2,
            "choices2": choices2,
            "rewards": rewards,
            "nll": nll,
        }
    return calc_fval(nll, params, prior=prior, output=output)


hybrid_mbmf_desc = """Hybrid model-based / model-free learner for the Daw et al.
(2011) two-step task (llm2b2alr.m). First-stage net values are a
weighted sum of model-based (Bellman) and model-free (SARSA(lambda)) values,
w*Q_MB + (1-w)*Q_MF, plus first-stage perseveration; choices at both stages are
softmax over these values. Nests the pure model-free (omega=0) and model-based
(omega=1) learners. Free parameters: beta1, beta2, alpha1, alpha2, lambda,
omega, r."""
hybrid_mbmf_id = "hybrid_mbmf"
hybrid_mbmf_spec = {
    "rl": {
        "softmax": ["beta1", "beta2"],
        "sarsa": ["alpha1", "alpha2", "lambda"],
        "weight": ["omega"],
        "choice": ["r"],
    }
}
hybrid_mbmf_model = ModelSpec(
    id=hybrid_mbmf_id, spec=hybrid_mbmf_spec, desc=hybrid_mbmf_desc.strip(),
    params=None, sim=hybrid_mbmf_sim, fit=hybrid_mbmf_fit,
)
