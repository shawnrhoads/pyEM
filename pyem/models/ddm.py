"""Drift-diffusion model (DDM) for a safe-vs-risky economic choice task.

On each trial the agent chooses between a SAFE bet (a certain, fixed amount)
and a RISKY bet (a gamble: win probability ``p``, payoff ``payoff``, so its
expected value is ``EV_risky = p * payoff``). We jointly model *which* option
was chosen and *how long* it took (response time, RT) with a two-boundary
Wiener diffusion process.

Boundary convention
-------------------
- UPPER boundary (absorbing at ``a``)  -> choose RISKY   (``choice = 1``)
- LOWER boundary (absorbing at ``0``)  -> choose SAFE    (``choice = 0``)
- absolute start point = ``z * a`` with relative bias ``z`` in ``(0, 1)``
- unit within-trial noise (``sigma = 1``)

The drift rate on a trial is driven by the risky-minus-safe value difference::

    v = v_coef * (EV_risky - safe)

so a positive drift pushes the accumulator toward the RISKY (upper) boundary.

Free parameters (4)
-------------------
- ``v_coef`` : drift scaling (real-valued, identity transform)
- ``a``      : boundary separation (>0)
- ``t0``     : non-decision time (>0, bounded transform, see ``t0_xform``)
- ``z``      : relative start-point bias in (0, 1)

Likelihood: Navarro & Fuss (2009) WFPT
--------------------------------------
:func:`wfpt_logpdf` returns the log first-passage-time density to the LOWER
boundary of a Wiener process with drift ``v``, boundary separation ``a`` and
relative start ``z``. Following Navarro & Fuss (2009), "Fast and accurate
calculations for first-passage times in Wiener diffusion models" (J. Math.
Psychol. 53:222-230), the density factorizes as::

    p(t | v, a, z) = (1 / a**2) * exp(-v*a*z - v**2 * t / 2) * f(t / a**2 | z)

where ``f(tt | z)`` is the first-passage density of the *driftless*, unit
boundary-separation process in normalized time ``tt = t / a**2``. ``f`` has two
convergent series representations, and we pick whichever needs fewer terms to
reach the tolerance (Navarro & Fuss term-count bounds):

- small-time (few terms when ``tt`` is small)::

      f = (2*pi*tt**3)**-0.5 * sum_k (z + 2k) * exp(-(z + 2k)**2 / (2*tt))

- large-time (few terms when ``tt`` is large)::

      f = pi * sum_{k>=1} k * exp(-k**2 * pi**2 * tt / 2) * sin(k*pi*z)

Both sums are assembled in LOG space via :func:`scipy.special.logsumexp` with
signed coefficients, so mixed-sign terms and tiny/large magnitudes stay stable.

The UPPER-boundary density (a RISKY choice) is obtained by the standard Wiener
reflection ``v -> -v``, ``z -> 1 - z``.

The observed RT includes the non-decision time, so the diffusion density is
evaluated at the decision time ``rt - t0``; ``rt <= t0`` has zero density
(returns ``-inf``).
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit, logsumexp
from ..utils.math import norm2alpha, norm2beta, calc_fval
from ..core.modelspec import ModelSpec

# Non-decision-time transform: t0 = T0_CAP * sigmoid(x - T0_SHIFT), in (0, T0_CAP).
# 0.5 s comfortably exceeds the sampled/plausible t0 range (~0.1-0.3 s) while
# keeping t0 strictly below realistic RTs. The T0_SHIFT offset is what makes the
# EM E-step usable: that optimizer starts every subject near raw x=0, and t0
# MUST start below the subject's fastest RT or the whole trial is infeasible
# (t0 >= rt -> flat 1e7 penalty the optimizer can't climb out of). The shift
# maps raw x=0 to t0 ~= 0.05 s, below any plausible RT, so the start is feasible
# for every subject; the sampled t0 in [0.1, 0.3] maps to raw x in ~[0.6, 2.4].
T0_CAP = 0.5
T0_SHIFT = 2.2

# Cap on the boundary-separation transform: a = A_CAP * sigmoid(x) in (0, A_CAP).
# This is deliberately *not* the generic norm2beta cap of 20: the E-step starts
# every subject near raw x=0, and with a 20-cap that maps to a=10, where the
# Wiener RTs are so long that every trial saturates the max-time floor and the
# likelihood is flat (no gradient -> the boundary never gets fit). A cap of 4
# maps raw x=0 to a=2, squarely inside the informative regime for the sampled
# a in [0.8, 2.0], so BFGS starts with real gradient signal on the boundary.
A_CAP = 4.0


def t0_xform(x):
    """Gaussian -> non-decision-time (0, ``T0_CAP``) via a shifted logistic.

    ``t0 = T0_CAP * sigmoid(x - T0_SHIFT)``. Named (not a lambda) so it can be
    referenced from the parameter registry and inspected/tested on its own. See
    ``T0_SHIFT`` for why the logistic is offset.
    """
    return T0_CAP * expit(np.asarray(x, dtype=float) - T0_SHIFT)


def a_xform(x):
    """Gaussian -> boundary separation (0, ``A_CAP``) via ``norm2beta`` with a
    reduced cap (see ``A_CAP`` for why the cap is 4, not the usual 20)."""
    return norm2beta(np.asarray(x, dtype=float), max_val=A_CAP)


# =============================================================================
# Wiener first-passage-time (WFPT) log density
# =============================================================================
def wfpt_logpdf(rt, v, a, z, err: float = 1e-10):
    """Log first-passage-time density to the **lower** boundary of a Wiener
    diffusion (Navarro & Fuss, 2009).

    Parameters
    ----------
    rt : float or array-like
        Decision time(s) (already net of non-decision time; must be > 0).
    v : float or array-like
        Drift rate(s). Broadcast against ``rt``.
    a : float
        Boundary separation (> 0). Scalar.
    z : float or array-like
        Relative start point in (0, 1) (absolute start = ``z * a``). Broadcast
        against ``rt``.
    err : float
        Target absolute error for the series truncation.

    Returns
    -------
    float or np.ndarray
        ``log p(rt | v, a, z)`` for absorption at the lower boundary. Returns
        ``-inf`` where ``rt <= 0``. Scalar in -> scalar out.

    Notes
    -----
    Get the UPPER-boundary density via ``wfpt_logpdf(rt, -v, a, 1 - z)``.
    """
    scalar_in = np.isscalar(rt) and np.isscalar(v) and np.isscalar(z)
    rt = np.atleast_1d(np.asarray(rt, dtype=float))
    v = np.broadcast_to(np.asarray(v, dtype=float), rt.shape)
    w = np.broadcast_to(np.asarray(z, dtype=float), rt.shape)
    a = float(a)
    a2 = a * a

    out = np.full(rt.shape, -np.inf, dtype=float)
    valid = rt > 0.0
    if not np.any(valid):
        return float(out[0]) if scalar_in else out

    rt_v = rt[valid]
    v_v = v[valid]
    w_v = w[valid]
    tt = rt_v / a2  # normalized time

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # ---- Navarro & Fuss (2009) number-of-terms bounds ----
        # large-time
        pit_err = np.pi * tt * err
        kl = np.where(
            pit_err < 1.0,
            np.maximum(
                np.sqrt(-2.0 * np.log(np.where(pit_err < 1.0, pit_err, 0.5))
                        / (np.pi ** 2 * tt)),
                1.0 / (np.pi * np.sqrt(tt)),
            ),
            1.0 / (np.pi * np.sqrt(tt)),
        )
        # small-time
        sq_err = 2.0 * np.sqrt(2.0 * np.pi * tt) * err
        ks = np.where(
            sq_err < 1.0,
            np.maximum(
                2.0 + np.sqrt(-2.0 * tt * np.log(np.where(sq_err < 1.0, sq_err, 0.5))),
                np.sqrt(tt) + 1.0,
            ),
            2.0,
        )

    use_small = ks < kl

    # --- large-time series (used where ~use_small) ---
    kl_needed = kl[~use_small]
    K_large = int(np.ceil(kl_needed.max())) if kl_needed.size else 1
    K_large = max(K_large, 1)
    kk = np.arange(1, K_large + 1, dtype=float)[None, :]      # (1, K_large)
    ang = np.pi * kk * w_v[:, None]                            # (n, K_large)
    sin_t = np.sin(ang)
    with np.errstate(divide="ignore"):
        log_abs_l = (np.log(kk)
                     + np.log(np.abs(sin_t))
                     - (kk ** 2) * (np.pi ** 2) * tt[:, None] / 2.0)
    lse_l, _ = logsumexp(log_abs_l, b=np.sign(sin_t), axis=1, return_sign=True)
    logf_large = lse_l + np.log(np.pi)

    # --- small-time series (used where use_small) ---
    ks_needed = ks[use_small]
    K_small = int(np.ceil(ks_needed.max())) if ks_needed.size else 2
    half = K_small // 2 + 1
    ks_range = np.arange(-half, half + 1, dtype=float)[None, :]  # symmetric, generous
    arg = w_v[:, None] + 2.0 * ks_range                          # (n, m)
    with np.errstate(divide="ignore"):
        log_abs_s = np.log(np.abs(arg)) - (arg ** 2) / (2.0 * tt[:, None])
    lse_s, _ = logsumexp(log_abs_s, b=np.sign(arg), axis=1, return_sign=True)
    logf_small = lse_s - 0.5 * np.log(2.0 * np.pi * tt ** 3)

    logf = np.where(use_small, logf_small, logf_large)

    # convert driftless normalized density -> full density (lower boundary)
    logp = logf - v_v * a * w_v - 0.5 * (v_v ** 2) * rt_v - np.log(a2)

    out[valid] = logp
    return float(out[0]) if scalar_in else out


# =============================================================================
# Simulation
# =============================================================================
def ddm_sim(params: np.ndarray, ntrials: int = 150, dt: float = 1e-3,
            max_time: float = 8.0, **kwargs) -> dict:
    """Simulate the safe-vs-risky DDM task by Euler-Maruyama integration.

    Parameters are supplied in **natural** space with columns
    ``[v_coef, a, t0, z]`` and shape ``(nsubjects, 4)``.

    Each trial draws a risky gamble (win prob ``p``, payoff) and a safe amount;
    ``EV_risky = p * payoff``. The drift is ``v = v_coef * (EV_risky - safe)``.
    The accumulator starts at ``z * a`` and steps
    ``X += v*dt + sqrt(dt) * N(0, 1)`` until it crosses ``a`` (upper -> risky)
    or ``0`` (lower -> safe); ``RT = steps*dt + t0``.

    Returns a dict with per-subject-per-trial arrays ``rt``, ``choice``
    (1 = risky/upper, 0 = safe/lower), ``ev_risky``, ``safe`` (each shape
    ``(nsubjects, ntrials)``), plus ``params`` and the drift ``v``.
    """
    params = np.asarray(params, dtype=float)
    if params.ndim != 2 or params.shape[1] != 4:
        raise ValueError("params must have shape (nsubjects, 4): [v_coef, a, t0, z]")

    nsubjects = params.shape[0]
    all_v_coef = params[:, 0]
    all_a = params[:, 1]
    all_t0 = params[:, 2]
    all_z = params[:, 3]

    if not ((all_a > 0.0) & (all_a <= A_CAP)).all():
        raise ValueError(f"a (boundary) out of bounds (0, {A_CAP}]")
    if not ((all_t0 >= 0.0) & (all_t0 < max_time)).all():
        raise ValueError("t0 out of bounds")
    if not ((all_z > 0.0) & (all_z < 1.0)).all():
        raise ValueError("z (bias) must be in (0, 1)")

    rng = np.random.default_rng()
    max_steps = int(np.ceil(max_time / dt))
    sqrt_dt = np.sqrt(dt)

    rt = np.zeros((nsubjects, ntrials), dtype=float)
    choice = np.zeros((nsubjects, ntrials), dtype=int)
    ev_risky = np.zeros((nsubjects, ntrials), dtype=float)
    safe = np.zeros((nsubjects, ntrials), dtype=float)
    v_all = np.zeros((nsubjects, ntrials), dtype=float)

    for s in range(nsubjects):
        v_coef = float(all_v_coef[s])
        a = float(all_a[s])
        t0 = float(all_t0[s])
        z = float(all_z[s])
        x0 = z * a

        # trial task variables: win prob, payoff, and a safe amount jittered
        # around the gamble's EV so the value difference spans +/- both ways.
        p_win = rng.uniform(0.1, 0.9, size=ntrials)
        payoff = rng.uniform(1.0, 4.0, size=ntrials)
        ev_r = p_win * payoff
        delta = rng.uniform(-1.2, 1.2, size=ntrials)   # = EV_risky - safe
        safe_amt = ev_r - delta

        ev_risky[s, :] = ev_r
        safe[s, :] = safe_amt

        for t in range(ntrials):
            v = v_coef * (ev_r[t] - safe_amt[t])
            v_all[s, t] = v
            x = x0
            step = 0
            while step < max_steps:
                x += v * dt + sqrt_dt * rng.standard_normal()
                step += 1
                if x >= a:
                    choice[s, t] = 1        # upper -> risky
                    break
                if x <= 0.0:
                    choice[s, t] = 0        # lower -> safe
                    break
            else:
                # timed out: assign by which boundary is closer
                choice[s, t] = 1 if x >= (a / 2.0) else 0
            rt[s, t] = step * dt + t0

    return {
        "params": np.column_stack([all_v_coef, all_a, all_t0, all_z]),
        "rt": rt,
        "choice": choice,
        "ev_risky": ev_risky,
        "safe": safe,
        "v": v_all,
    }


# =============================================================================
# Fit
# =============================================================================
def ddm_fit(params, rt, choice, ev_risky, safe, prior=None, output: str = "npl"):
    """EM-compatible objective for the safe-vs-risky DDM.

    ``params`` are in **normalized** (Gaussian) space with columns
    ``[v_coef, a, t0, z]`` decoded as::

        v_coef = params[0]                 # identity (real-valued)
        a      = a_xform(params[1])        # -> (0, A_CAP)
        t0     = t0_xform(params[2])       # -> (0, T0_CAP)
        z      = norm2alpha(params[3])     # -> (0, 1)

    Returns ``1e7`` when ``a``/``t0``/``z`` fall outside their natural bounds,
    or when any observed ``rt <= t0`` (implying a non-positive decision time).
    """
    v_coef = float(params[0])
    a = float(a_xform(params[1]))
    t0 = float(t0_xform(params[2]))
    z = float(norm2alpha(params[3]))

    rt = np.asarray(rt, dtype=float).ravel()
    choice = np.asarray(choice).ravel().astype(int)
    ev_risky = np.asarray(ev_risky, dtype=float).ravel()
    safe = np.asarray(safe, dtype=float).ravel()

    # An out-of-bounds parameter, or a non-decision time so large that some
    # trial has a non-positive decision time (t0 >= rt), makes the model
    # undefined. Enforce it as a hard 1e7 penalty on the OPTIMIZATION path (so
    # the MLE of t0 stays below min(rt), as it must), but still return a
    # well-formed dict for output="all" (used by EMModel.get_outfit) by
    # evaluating on clipped decision times.
    infeasible = (
        not (-20.0 <= v_coef <= 20.0)
        or not (1e-5 <= a <= A_CAP)
        or not (0.0 < t0 < T0_CAP)
        or not (0.0 < z < 1.0)
        or bool(np.any(rt - t0 <= 0.0))
    )
    if infeasible and output != "all":
        return 1e7

    dt = np.clip(rt - t0, 1e-6, None)       # decision time (clipped only for bookkeeping)

    v = v_coef * (ev_risky - safe)          # per-trial drift

    # lower boundary (safe, choice==0): wfpt_logpdf(dt, v, a, z)
    # upper boundary (risky, choice==1): reflection v->-v, z->1-z
    is_risky = choice == 1
    v_eff = np.where(is_risky, -v, v)
    z_eff = np.where(is_risky, 1.0 - z, z)

    logdens = np.asarray(wfpt_logpdf(dt, v_eff, a, z_eff), dtype=float)
    # Floor rare underflow / timed-out trials (density ~ 0 -> -inf) to a large
    # finite penalty so a single extreme RT can't invalidate an otherwise good
    # parameter set (and keeps NLL finite for get_outfit).
    logdens = np.where(np.isfinite(logdens), logdens, -1e3)

    nll = float(-np.sum(logdens))

    if output == "all":
        return {
            "params": [v_coef, a, t0, z],
            "rt": rt,
            "choice": choice,
            "ev_risky": ev_risky,
            "safe": safe,
            "v": v,
            "logdens": logdens,
            "nll": nll,
        }

    return calc_fval(nll, params, prior=prior, output=output)


ddm_desc = """Drift-diffusion model (DDM) of a safe-vs-risky economic choice.
Each trial pits a SAFE certain amount against a RISKY gamble (EV = p*payoff);
both the choice and the response time are modelled by a two-boundary Wiener
diffusion (upper = risky, lower = safe; start = z*a). Drift is driven by the
risky-minus-safe value difference: v = v_coef * (EV_risky - safe). The
likelihood is the Navarro & Fuss (2009) first-passage-time density evaluated in
log space at the decision time rt - t0.
Free parameters: v_coef (drift scaling), a (boundary separation),
t0 (non-decision time), z (start-point bias)."""
ddm_id = "ddm"
ddm_spec = {
    "ddm": {"drift": ["v_coef"], "boundary": ["a"], "ndt": ["t0"], "bias": ["z"]},
    "likelihood": "navarro_fuss_2009_wfpt",
    "boundaries": {"upper": "risky", "lower": "safe"},
}
ddm_model = ModelSpec(
    id=ddm_id, spec=ddm_spec, desc=ddm_desc.strip(),
    params=None, sim=ddm_sim, fit=ddm_fit,
)
