"""Drift-diffusion models (DDM) for two value-based choice tasks, each in a
four-parameter and a seven-parameter ("full DDM") form — four models sharing a
single Wiener first-passage-time (WFPT) likelihood.

Two tasks
---------
1. **High-vs-low value** (``ddm4`` / ``ddm7``): on each trial the agent is
   offered two *certain* amounts and should choose the higher-valued one. The
   drift is driven by the value gap and points toward the UPPER (correct/high)
   boundary, so the task has a well-defined accuracy: ``value_high`` and
   ``value_low`` with ``value_high > value_low``, drift
   ``v = v_coef * (value_high - value_low) >= 0``. UPPER = chose HIGH (correct),
   LOWER = chose LOW (error).
2. **Safe-vs-risky gamble** (``ddm4_lotto`` / ``ddm7_lotto``): on each trial a
   RISKY gamble (win probability ``p``, payoff ``payoff``, so
   ``EV_risky = p * payoff``) is pitted against a SAFE certain amount. Drift is
   the risky-minus-safe value difference ``v = v_coef * (EV_risky - safe)`` and
   can point either way. UPPER = RISKY (``choice = 1``), LOWER = SAFE
   (``choice = 0``).

Two parameterizations (per task)
--------------------------------
- **Four-parameter** (``ddm4`` / ``ddm4_lotto``): ``[v_coef, a, t0, z]`` — drift
  scaling, boundary separation, non-decision time, relative start-point bias.
- **Seven-parameter** (``ddm7`` / ``ddm7_lotto``): adds the three across-trial
  variability parameters of the "full" diffusion model (Ratcliff & Rouder,
  1998; Ratcliff & Tuerlinckx, 2002; Henrich, Hartmann, Pratz, Voss & Klauer,
  2024, *Behav. Res. Methods* 56:3102-3116):

  - ``sv`` : SD of a Normal on the trial drift,
    ``v_trial ~ Normal(v_mean, sv)`` with ``sv >= 0``.
  - ``st`` : **full width** of a Uniform on the non-decision time,
    ``t0_trial ~ U(t0 - st/2, t0 + st/2)`` with ``st >= 0``.
  - ``sz`` : **full width** of a Uniform on the **relative** start point,
    ``z_trial ~ U(z - sz/2, z + sz/2)``, interval constrained to ``(0, 1)``.

  Setting ``sv = st = sz = 0`` reduces each seven-parameter model **exactly** to
  its four-parameter sibling on the same task.

Boundary convention (both tasks)
--------------------------------
UPPER boundary (at ``a``) = ``choice = 1``, LOWER boundary (at ``0``) =
``choice = 0``, absolute start point ``z * a``, unit within-trial noise.

Likelihood (shared by all four models)
--------------------------------------
The drift variability ``sv`` is marginalized **analytically** in closed form
(Ratcliff & Tuerlinckx, 2002): with drift ``v ~ Normal(v, sv)`` the
lower-boundary density is

    log p(rt | v, a, z, sv)
        = log f(rt / a^2 | z) - log(a^2) - 0.5 * log(sv^2 * rt + 1)
          + (a^2 * z^2 * sv^2 - 2*a*z*v - v^2*rt) / (2 * (sv^2 * rt + 1))

where ``f(tt | z)`` is the driftless normalized first-passage density
(Navarro & Fuss, 2009). The remaining variabilities ``st`` and ``sz`` are
integrated numerically by fixed-order Gauss-Legendre quadrature over the uniform
intervals; the order is exposed via ``*_fit(..., n_st=, n_sz=)`` on the
seven-parameter fits. All four fits get the UPPER-boundary density by the
standard reflection ``v -> -v``, ``z -> 1 - z``.

Shared implementation
---------------------
- Likelihood: :func:`_wfpt_logf`, :func:`wfpt_sv_logpdf`, :func:`_marginal_logpdf`,
  :func:`wfpt_logpdf` (the ``sv = 0`` special case).
- Numeric engine: :func:`_simulate_diffusion` (the vectorized Euler-Maruyama
  first-passage integrator), :func:`_wfpt_nll` (boundary reflection + summed log
  density), and :func:`_simulate_paths` (trajectory recorder for visualization).
- Everything task- and parameterization-specific — task-variable generation, the
  drift equation, the free-parameter set, bounds, and feasibility rules — lives
  explicitly in each model's own ``*_sim`` / ``*_fit``.

Recoverability note (seven-parameter variability)
--------------------------------------------------
A naive per-trial Euler-Maruyama simulator at ``dt = 1e-3`` biases the data: the
boundary-overshoot bias (~``0.58*sqrt(dt)``) exceeds the likelihood signal of
plausible ``sv`` values, so maximum-likelihood ``sv`` collapses to 0 and ``sz``
is biased low. The simulators here run at ``dt = 1e-4`` (vectorized, so faster
than a per-trial loop despite the finer grid), which removes that bias (verified
by profile-likelihood tests in ``tests/test_ddm.py``). Even so, a 50-subject x
500-trial hierarchical EM recovery study found the variability parameters only
weakly identified in these value-based designs (achieved Pearson r for the
gamble task: v_coef .81, a .95, t0 .83, z .98, sv .17, st .58, sz -.04),
consistent with the literature requiring specialized designs/trial counts. The
four-parameter models (``ddm4`` / ``ddm4_lotto``) are therefore recommended for
parameter recovery; the seven-parameter models are retained for likelihood
evaluation and for simulating richer generative designs.

References
----------
- Navarro & Fuss (2009). *J. Math. Psychol.* 53:222-230.
- Ratcliff & Tuerlinckx (2002). *Psychon. Bull. Rev.* 9:438-481.
- Ratcliff & McKoon (2008). *Neural Computation* 20:873-922.
- Henrich, Hartmann, Pratz, Voss & Klauer (2024). *Behav. Res. Methods*
  56:3102-3116.
- Hartmann & Klauer (2021). *J. Math. Psychol.* 103:102550.
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit, logsumexp, roots_legendre
from ..utils.math import norm2alpha, norm2beta, calc_fval
from ..core.modelspec import ModelSpec

_DDM7_NOT_SUPPORTED = (
    "The seven-parameter DDM (ddm7 / ddm7_lotto) is not supported in this "
    "release yet. Use ddm4 / ddm4_lotto."
)

# --- Core-parameter transforms shared by all four models ---
T0_CAP = 0.5
T0_SHIFT = 2.2
A_CAP = 4.0


def t0_xform(x):
    """Gaussian -> non-decision-time (0, ``T0_CAP``) via a shifted logistic,
    ``t0 = T0_CAP * sigmoid(x - T0_SHIFT)``."""
    return T0_CAP * expit(np.asarray(x, dtype=float) - T0_SHIFT)


def a_xform(x):
    """Gaussian -> boundary separation (0, ``A_CAP``) via ``norm2beta`` with a
    reduced cap."""
    return norm2beta(np.asarray(x, dtype=float), max_val=A_CAP)


# --- Across-trial variability transforms (seven-parameter models) ---
# Caps sized to the Henrich et al. (2024) regime (their ground truth: sv ~
# N(1,3) truncated [0,3]; st0 ~ N(0.183,0.09) truncated [0,0.5]; sw ~ B(1,3));
# a cap of 1 cannot even represent typical sv values in this regime.
SV_CAP = 2.5
ST_CAP = 0.4
SZ_CAP = 0.9
_EPS_VAR = 1e-8   # a variability below this collapses its integration dimension to a point


def sv_xform(x):
    """Gaussian -> drift SD (0, SV_CAP)."""
    return norm2beta(np.asarray(x, dtype=float), max_val=SV_CAP)


def st_xform(x):
    """Gaussian -> non-decision-time uniform full width (0, ST_CAP)."""
    return norm2beta(np.asarray(x, dtype=float), max_val=ST_CAP)


def sz_xform(x):
    """Gaussian -> relative-start uniform full width (0, SZ_CAP)."""
    return norm2beta(np.asarray(x, dtype=float), max_val=SZ_CAP)


# =============================================================================
# Shared likelihood: driftless normalized WFPT log density (Navarro & Fuss, 2009)
# =============================================================================
def _wfpt_logf(tt, w, err: float = 1e-10):
    """Driftless normalized log first-passage density ``log f(tt | w)``.

    ``tt`` (normalized time = rt / a^2) and ``w`` (relative start) are
    equal-shape 1-D arrays; entries of ``tt`` are assumed strictly positive.
    Returns the same-shape ``logf``.
    """
    tt = np.asarray(tt, dtype=float)
    w = np.asarray(w, dtype=float)

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
    ang = np.pi * kk * w[:, None]                             # (n, K_large)
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
    arg = w[:, None] + 2.0 * ks_range                            # (n, m)
    with np.errstate(divide="ignore"):
        log_abs_s = np.log(np.abs(arg)) - (arg ** 2) / (2.0 * tt[:, None])
    lse_s, _ = logsumexp(log_abs_s, b=np.sign(arg), axis=1, return_sign=True)
    logf_small = lse_s - 0.5 * np.log(2.0 * np.pi * tt ** 3)

    logf = np.where(use_small, logf_small, logf_large)
    return logf


# =============================================================================
# Shared likelihood: analytic drift-variability (sv) density (Ratcliff & Tuerlinckx, 2002)
# =============================================================================
def wfpt_sv_logpdf(rt, v, a, z, sv=0.0):
    """Log **lower**-boundary first-passage density with drift ~ Normal(v, sv),
    integrated analytically over the drift (Ratcliff & Tuerlinckx, 2002).

    Vectorized over ``rt``/``v``/``z`` (broadcast against ``rt``); ``a`` and
    ``sv`` are scalars. Returns ``-inf`` where ``rt <= 0``. At ``sv = 0`` this
    reduces **exactly** to :func:`wfpt_logpdf`. Get the UPPER-boundary density
    via the reflection ``v -> -v``, ``z -> 1 - z``. Validated against dense
    numerical integration of the base density over ``v' ~ Normal(v, sv)`` in
    ``tests/test_ddm.py``.
    """
    scalar_in = np.isscalar(rt) and np.isscalar(v) and np.isscalar(z)
    rt = np.atleast_1d(np.asarray(rt, dtype=float))
    v = np.broadcast_to(np.asarray(v, dtype=float), rt.shape)
    w = np.broadcast_to(np.asarray(z, dtype=float), rt.shape)
    a = float(a)
    sv = float(sv)
    a2 = a * a

    out = np.full(rt.shape, -np.inf, dtype=float)
    valid = rt > 0.0
    if not np.any(valid):
        return float(out[0]) if scalar_in else out

    rt_v = rt[valid]
    v_v = v[valid]
    w_v = w[valid]
    tt = rt_v / a2

    logf = _wfpt_logf(tt, w_v)

    sv2 = sv * sv
    denom = sv2 * rt_v + 1.0
    logp = (logf
            - np.log(a2)
            - 0.5 * np.log(denom)
            + (a2 * w_v ** 2 * sv2 - 2.0 * a * w_v * v_v - v_v ** 2 * rt_v)
            / (2.0 * denom))

    out[valid] = logp
    return float(out[0]) if scalar_in else out


# =============================================================================
# Shared likelihood: numerical marginal over st (non-decision) and sz (start)
# =============================================================================
def _marginal_logpdf(rt, v, a, zc, t0, sv, st, sz, n_st=11, n_sz=11):
    """Log density marginalized over ``t0_trial ~ U(t0 +/- st/2)`` and
    ``z_trial ~ U(zc +/- sz/2)`` by Gauss-Legendre quadrature; drift variability
    ``sv`` is handled analytically inside :func:`wfpt_sv_logpdf`.

    ``rt``, ``v`` and ``zc`` are per-trial 1-D arrays of length ``n`` (``zc`` is
    the per-trial relative-start CENTRE, which may be the reflected value
    ``1 - z`` on upper-boundary trials). Returns a length-``n`` array of log
    densities. Decision times ``rt - tau <= 0`` return ``-inf`` (truncation
    handled automatically while still dividing by the full ``st`` via the fixed
    weights).
    """
    rt = np.asarray(rt, dtype=float).ravel()
    n = rt.shape[0]
    v = np.broadcast_to(np.asarray(v, dtype=float).ravel(), (n,))
    zc = np.broadcast_to(np.asarray(zc, dtype=float).ravel(), (n,))

    # --- non-decision-time nodes tau over U(t0 +/- st/2) ---
    if st < _EPS_VAR:
        tau = np.array([t0], dtype=float)
        logw_t = np.array([0.0])
    else:
        x_t, w_t = roots_legendre(n_st)
        tau = t0 + (st / 2.0) * x_t
        logw_t = np.log(w_t / 2.0)
    n_tau = tau.shape[0]

    # --- relative-start nodes z' over U(zc +/- sz/2) (per-trial) ---
    if sz < _EPS_VAR:
        zgrid = zc[:, None]                      # (n, 1)
        logw_z = np.array([0.0])
    else:
        x_z, w_z = roots_legendre(n_sz)
        logw_z = np.log(w_z / 2.0)
        zgrid = zc[:, None] + (sz / 2.0) * x_z[None, :]   # (n, n_sz)
    n_z = zgrid.shape[1]

    # --- build the (n, n_tau, n_z) grid ---
    dt = rt[:, None, None] - tau[None, :, None]           # (n, n_tau, 1)
    dt = np.broadcast_to(dt, (n, n_tau, n_z))
    zg = np.broadcast_to(zgrid[:, None, :], (n, n_tau, n_z))
    vg = np.broadcast_to(v[:, None, None], (n, n_tau, n_z))

    logdens = wfpt_sv_logpdf(dt.ravel(), vg.ravel(), a, zg.ravel(), sv)
    logdens = np.asarray(logdens, dtype=float).reshape(n, n_tau, n_z)

    terms = logdens + logw_t[None, :, None] + logw_z[None, None, :]
    return logsumexp(terms.reshape(n, -1), axis=1)


def wfpt_logpdf(rt, v, a, z):
    """Log lower-boundary WFPT density (Navarro & Fuss, 2009) — the ``sv = 0``
    special case of :func:`wfpt_sv_logpdf`, exposed as a public convenience for
    callers who want the base-model density directly. Get the UPPER-boundary
    density via the reflection ``v -> -v``, ``z -> 1 - z``.
    """
    return wfpt_sv_logpdf(rt, v, a, z, sv=0.0)


# =============================================================================
# Shared numeric engine: forward simulation and the fit objective
# =============================================================================
def _simulate_diffusion(v_draw, a, z_draw, t0_draw, dt, max_time, rng):
    """Vectorized Euler-Maruyama first-passage engine shared by all four DDM
    simulators.

    Given already-drawn per-trial drift (``v_draw``), boundary separation
    (``a``), absolute-start-relative bias (``z_draw``) and non-decision time
    (``t0_draw``) — all equal-length 1-D arrays over the flattened trials —
    integrate ``x += v*dt + sqrt(dt)*N(0,1)`` from ``x0 = z_draw * a`` until it
    crosses ``a`` (upper) or ``0`` (lower). Returns flat arrays
    ``(rt, choice, crossed)``: ``choice`` 1 = upper, 0 = lower; timed-out trials
    are assigned to the nearer boundary and flagged ``crossed = False``.
    """
    n = v_draw.shape[0]
    x = z_draw * a
    choice = np.zeros(n, dtype=int)
    steps = np.zeros(n, dtype=int)
    active = np.ones(n, dtype=bool)
    max_steps = int(np.ceil(max_time / dt))
    sqrt_dt = np.sqrt(dt)

    for s in range(max_steps):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        x[idx] += v_draw[idx] * dt + sqrt_dt * rng.standard_normal(idx.size)
        steps[idx] = s + 1
        xi = x[idx]
        up = xi >= a[idx]
        dn = xi <= 0.0
        choice[idx[up]] = 1          # upper boundary
        active[idx[up]] = False
        active[idx[dn]] = False      # lower boundary (choice stays 0)
    crossed = ~active
    # timed-out trials: assign by which boundary is closer
    to = np.flatnonzero(active)
    choice[to] = (x[to] >= a[to] / 2.0).astype(int)

    rt = steps * dt + t0_draw
    return rt, choice, crossed


def _wfpt_nll(rt, choice, v_mean, a, t0, z, sv, st, sz, n_st=11, n_sz=11):
    """Summed negative log WFPT density given per-trial mean drift ``v_mean``
    and decoded scalar parameters — shared by all four DDM fit functions.

    Applies the boundary reflection (an UPPER-boundary hit, ``choice == 1``,
    uses ``v -> -v`` and ``z -> 1 - z``), marginalizes ``sv``/``st``/``sz`` via
    :func:`_marginal_logpdf`, floors non-finite per-trial log densities to
    ``-1e3``, and returns ``(nll, logdens)``.
    """
    is_upper = choice == 1
    v_eff = np.where(is_upper, -v_mean, v_mean)
    zc = np.where(is_upper, 1.0 - z, z)

    logdens = _marginal_logpdf(rt, v_eff, a, zc, t0, sv, st, sz,
                               n_st=n_st, n_sz=n_sz)
    logdens = np.asarray(logdens, dtype=float)
    logdens = np.where(np.isfinite(logdens), logdens, -1e3)
    nll = float(-np.sum(logdens))
    return nll, logdens


def _simulate_paths(v_draw, a, z_draw, t0_draw, dt, max_time, rng):
    """Per-trial evidence-trajectory recorder shared by the ``*_sim_paths``
    visualization helpers.

    ``a`` is a scalar; ``v_draw``, ``z_draw`` and ``t0_draw`` are per-trial 1-D
    arrays. Returns ``(ts, xs, choice, crossed, rt)`` where ``ts``/``xs`` are
    lists of equal-length time/evidence arrays per trial (the time axis is
    offset by the trial's non-decision time, and the final evidence point sits
    exactly on the boundary for crossed trials).
    """
    ntrials = v_draw.shape[0]
    max_steps = int(np.ceil(max_time / dt))
    sqrt_dt = np.sqrt(dt)

    ts, xs = [], []
    choice = np.zeros(ntrials, dtype=int)
    crossed = np.zeros(ntrials, dtype=bool)
    rt = np.zeros(ntrials, dtype=float)

    for i in range(ntrials):
        x = np.empty(max_steps + 1, dtype=float)
        x[0] = z_draw[i] * a
        n_used = max_steps
        for s in range(1, max_steps + 1):
            x[s] = x[s - 1] + v_draw[i] * dt + sqrt_dt * rng.standard_normal()
            if x[s] >= a:
                x[s] = a
                choice[i] = 1
                crossed[i] = True
                n_used = s
                break
            if x[s] <= 0.0:
                x[s] = 0.0
                choice[i] = 0
                crossed[i] = True
                n_used = s
                break
        else:
            # timed out: assign by which boundary is closer (as in the sims)
            choice[i] = 1 if x[max_steps] >= (a / 2.0) else 0
        rt[i] = n_used * dt + t0_draw[i]
        ts.append(t0_draw[i] + dt * np.arange(n_used + 1))
        xs.append(x[:n_used + 1].copy())

    return ts, xs, choice, crossed, rt


def _validate_core(all_a, all_t0, all_z, max_time):
    """Shared natural-space validation for the core parameters a, t0, z."""
    if not ((all_a > 0.0) & (all_a <= A_CAP)).all():
        raise ValueError(f"a (boundary) out of bounds (0, {A_CAP}]")
    if not ((all_t0 >= 0.0) & (all_t0 < max_time)).all():
        raise ValueError("t0 out of bounds")
    if not ((all_z > 0.0) & (all_z < 1.0)).all():
        raise ValueError("z (bias) must be in (0, 1)")


def _validate_variability(all_sv, all_st, all_sz, all_z):
    """Shared natural-space validation for the variability parameters
    sv, st, sz (seven-parameter simulators)."""
    if not (all_sv >= 0.0).all():
        raise ValueError("sv (drift SD) must be >= 0")
    if not (all_st >= 0.0).all():
        raise ValueError("st (non-decision-time width) must be >= 0")
    if not (all_sz >= 0.0).all():
        raise ValueError("sz (start-point width) must be >= 0")
    if not (((all_z - all_sz / 2.0) > 0.0) & ((all_z + all_sz / 2.0) < 1.0)).all():
        raise ValueError("start-point interval z +/- sz/2 must lie in (0, 1)")


def _draw_highlow_task(n, rng):
    """High-vs-low VALUE task: draw two certain amounts per trial with
    ``value_high > value_low`` (a low base plus a positive gap), so difficulty =
    the value gap and the correct choice is always the HIGH (upper) option.
    Returns ``(value_high, value_low)``."""
    value_low = rng.uniform(1.0, 3.0, size=n)
    value_gap = rng.uniform(0.1, 1.5, size=n)     # value_high - value_low > 0
    return value_low + value_gap, value_low


def _draw_lotto_task(n, rng):
    """Safe-vs-risky GAMBLE task: a risky gamble (win prob in [0.1, 0.9],
    payoff in [1, 4], so ``EV_risky = p * payoff``) versus a safe certain amount
    jittered around the gamble's EV so the value difference spans +/- both ways.
    Returns ``(ev_risky, safe)``."""
    p_win = rng.uniform(0.1, 0.9, size=n)
    payoff = rng.uniform(1.0, 4.0, size=n)
    ev_risky = p_win * payoff
    delta = rng.uniform(-1.2, 1.2, size=n)        # = EV_risky - safe
    return ev_risky, ev_risky - delta


# =============================================================================
# TASK 1 --- High-vs-low VALUE choice (deterministic, no gamble)
# =============================================================================
def ddm4_sim(params: np.ndarray, ntrials: int = 150, dt: float = 1e-4,
             max_time: float = 8.0, rng=None, **kwargs) -> dict:
    """Simulate the **four-parameter** high-vs-low VALUE task.

    Natural-space columns ``[v_coef, a, t0, z]``, shape ``(nsubjects, 4)``. Each
    trial offers two certain amounts (``value_high > value_low``); the agent
    should choose the higher one. Drift ``v = v_coef * (value_high - value_low)``
    is non-negative and points at the UPPER (correct/HIGH) boundary; the start is
    ``z * a`` with unit noise. ``RT = steps*dt + t0``. Returns per-trial arrays
    ``rt``, ``choice`` (1 = HIGH/upper/correct, 0 = LOW/lower/error),
    ``value_high``, ``value_low``, and ``v`` (per-trial mean drift), each shaped
    ``(nsubjects, ntrials)``, plus ``params``.
    """
    params = np.asarray(params, dtype=float)
    if params.ndim != 2 or params.shape[1] != 4:
        raise ValueError("params must have shape (nsubjects, 4): [v_coef, a, t0, z]")
    nsubjects = params.shape[0]
    all_v_coef, all_a, all_t0, all_z = (params[:, i] for i in range(4))
    _validate_core(all_a, all_t0, all_z, max_time)
    if rng is None:
        rng = np.random.default_rng()

    n = nsubjects * ntrials
    v_coef = np.repeat(all_v_coef, ntrials)
    a = np.repeat(all_a, ntrials)
    z = np.repeat(all_z, ntrials)
    t0 = np.repeat(all_t0, ntrials)

    value_high, value_low = _draw_highlow_task(n, rng)
    v_mean = v_coef * (value_high - value_low)    # >= 0, drift toward HIGH (upper)

    rt, choice, _ = _simulate_diffusion(v_mean, a, z, t0, dt, max_time, rng)

    shape = (nsubjects, ntrials)
    return {
        "params": np.column_stack([all_v_coef, all_a, all_t0, all_z]),
        "rt": rt.reshape(shape),
        "choice": choice.reshape(shape),
        "value_high": value_high.reshape(shape),
        "value_low": value_low.reshape(shape),
        "v": v_mean.reshape(shape),
    }


def ddm4_fit(params, rt, choice, value_high, value_low, prior=None,
             output: str = "npl"):
    """EM-compatible objective for the **four-parameter** high-vs-low VALUE task.

    Normalized columns ``[v_coef, a, t0, z]`` decoded ``v_coef = params[0]``
    (identity), ``a = a_xform(params[1])``, ``t0 = t0_xform(params[2])``,
    ``z = norm2alpha(params[3])``. Drift ``v = v_coef * (value_high - value_low)``;
    ``choice`` 1 = HIGH (upper), 0 = LOW (lower). With no non-decision-time
    spread, a ``t0`` at or above ANY observed ``rt`` makes the model undefined,
    so ``np.any(rt - t0 <= 0)`` is a hard ``1e7`` penalty on the optimization
    path; ``output="all"`` still returns a well-formed dict.
    """
    v_coef = float(params[0])
    a = float(a_xform(params[1]))
    t0 = float(t0_xform(params[2]))
    z = float(norm2alpha(params[3]))

    rt = np.asarray(rt, dtype=float).ravel()
    choice = np.asarray(choice).ravel().astype(int)
    value_high = np.asarray(value_high, dtype=float).ravel()
    value_low = np.asarray(value_low, dtype=float).ravel()

    infeasible = (
        not (-20.0 <= v_coef <= 20.0)
        or not (1e-5 <= a <= A_CAP)
        or not (0.0 < t0 < T0_CAP)
        or not (0.0 < z < 1.0)
        or bool(np.any(rt - t0 <= 0.0))
    )
    if infeasible and output != "all":
        return 1e7

    v_mean = v_coef * (value_high - value_low)   # drift toward HIGH (upper)
    nll, logdens = _wfpt_nll(rt, choice, v_mean, a, t0, z, 0.0, 0.0, 0.0)

    if output == "all":
        return {
            "params": [v_coef, a, t0, z],
            "rt": rt, "choice": choice,
            "value_high": value_high, "value_low": value_low,
            "v": v_mean, "logdens": logdens, "nll": nll,
        }
    return calc_fval(nll, params, prior=prior, output=output)


def ddm4_sim_paths(params, ntrials: int = 25, dt: float = 1e-3,
                   max_time: float = 4.0, rng=None) -> dict:
    """Simulate a SMALL number of trials of the **four-parameter** high-vs-low
    VALUE model while recording the full evidence trajectory of every trial, for
    visualization.

    ``params`` is a single subject's natural-space vector ``[v_coef, a, t0, z]``
    (shape ``(4,)`` or ``(1, 4)``). There is no across-trial variability, so each
    trial's drift is the value gap ``v_coef * (value_high - value_low)`` and the
    start point (``z``) and non-decision time (``t0``) are constant across trials.
    The default ``dt = 1e-3`` is coarser than ``ddm4_sim``'s ``1e-4`` because the
    purpose is drawing legible paths, not unbiased RT distributions. Returns, per
    trial: ``t`` (time arrays offset by ``t0``), ``x`` (evidence arrays), plus
    ``choice`` (1 = HIGH/upper, 0 = LOW/lower), ``crossed``, ``rt``, ``v`` (mean
    drift), ``v_draw``, ``value_high``, ``value_low``, ``t0_draw``, ``z_draw`` and
    ``params``.
    """
    params = np.asarray(params, dtype=float).reshape(-1)
    if params.shape[0] != 4:
        raise ValueError(
            "params must be a single subject's 4-vector: [v_coef, a, t0, z]")
    v_coef, a, t0, z = (float(p) for p in params)
    _validate_core(np.array([a]), np.array([t0]), np.array([z]), max_time)
    if rng is None:
        rng = np.random.default_rng()

    value_high, value_low = _draw_highlow_task(ntrials, rng)
    v_mean = v_coef * (value_high - value_low)
    v_draw = v_mean                       # no drift variability (sv = 0)
    z_draw = np.full(ntrials, z)          # no start-point variability (sz = 0)
    t0_draw = np.full(ntrials, t0)        # no non-decision variability (st = 0)

    ts, xs, choice, crossed, rt = _simulate_paths(
        v_draw, a, z_draw, t0_draw, dt, max_time, rng)

    return {
        "params": params.copy(),
        "t": ts, "x": xs,
        "choice": choice, "crossed": crossed, "rt": rt,
        "v": v_mean, "v_draw": v_draw,
        "value_high": value_high, "value_low": value_low,
        "t0_draw": t0_draw, "z_draw": z_draw,
    }


def ddm7_sim(params: np.ndarray, ntrials: int = 150, dt: float = 1e-4,
             max_time: float = 8.0, rng=None, **kwargs) -> dict:
    """Simulate the **seven-parameter** high-vs-low VALUE task.

    Natural-space columns ``[v_coef, a, t0, z, sv, st, sz]``, shape
    ``(nsubjects, 7)``. Same high-vs-low task as :func:`ddm4_sim`, but the trial
    drift, non-decision time and relative start are drawn from their
    across-trial distributions ``v_draw ~ Normal(v_mean, sv)``,
    ``t0_draw ~ U(t0 +/- st/2)``, ``z_draw ~ U(z +/- sz/2)`` with
    ``v_mean = v_coef * (value_high - value_low)``. Returns the same keys as
    :func:`ddm4_sim` plus ``sv``, ``st``, ``sz`` broadcast to
    ``(nsubjects, ntrials)``.
    """
    raise NotImplementedError(_DDM7_NOT_SUPPORTED)
    # --- disabled in this release; preserved for future support ---
    # params = np.asarray(params, dtype=float)
    # if params.ndim != 2 or params.shape[1] != 7:
    #     raise ValueError(
    #         "params must have shape (nsubjects, 7): [v_coef, a, t0, z, sv, st, sz]")
    # nsubjects = params.shape[0]
    # all_v_coef, all_a, all_t0, all_z, all_sv, all_st, all_sz = (
    #     params[:, i] for i in range(7))
    # _validate_core(all_a, all_t0, all_z, max_time)
    # _validate_variability(all_sv, all_st, all_sz, all_z)
    # if rng is None:
    #     rng = np.random.default_rng()
    #
    # n = nsubjects * ntrials
    # v_coef = np.repeat(all_v_coef, ntrials)
    # a = np.repeat(all_a, ntrials)
    # z = np.repeat(all_z, ntrials)
    # t0 = np.repeat(all_t0, ntrials)
    # sv = np.repeat(all_sv, ntrials)
    # st = np.repeat(all_st, ntrials)
    # sz = np.repeat(all_sz, ntrials)
    #
    # value_high, value_low = _draw_highlow_task(n, rng)
    # v_mean = v_coef * (value_high - value_low)    # >= 0, drift toward HIGH (upper)
    # v_draw = v_mean + sv * rng.standard_normal(n)
    # z_draw = z + sz * (rng.random(n) - 0.5)
    # t0_draw = t0 + st * (rng.random(n) - 0.5)
    #
    # rt, choice, _ = _simulate_diffusion(v_draw, a, z_draw, t0_draw, dt, max_time, rng)
    #
    # shape = (nsubjects, ntrials)
    # return {
    #     "params": np.column_stack(
    #         [all_v_coef, all_a, all_t0, all_z, all_sv, all_st, all_sz]),
    #     "rt": rt.reshape(shape),
    #     "choice": choice.reshape(shape),
    #     "value_high": value_high.reshape(shape),
    #     "value_low": value_low.reshape(shape),
    #     "v": v_mean.reshape(shape),
    #     "sv": sv.reshape(shape).copy(),
    #     "st": st.reshape(shape).copy(),
    #     "sz": sz.reshape(shape).copy(),
    # }


def ddm7_fit(params, rt, choice, value_high, value_low, prior=None,
             output: str = "npl", n_st: int = 11, n_sz: int = 11):
    """EM-compatible objective for the **seven-parameter** high-vs-low VALUE task.

    Normalized columns ``[v_coef, a, t0, z, sv, st, sz]`` with the first four
    decoded as in :func:`ddm4_fit` and ``sv = sv_xform(params[4])``,
    ``st = st_xform(params[5])``, ``sz = sz_xform(params[6])``. Drift
    ``v = v_coef * (value_high - value_low)``; ``choice`` 1 = HIGH (upper).
    ``n_st``/``n_sz`` set the Gauss-Legendre order. Because ``t0`` is spread over
    ``U(t0 +/- st/2)``, infeasible non-decision draws are handled per trial inside
    :func:`_marginal_logpdf` (no hard ``rt <= t0`` rejection); out-of-bounds
    parameters (or ``z +/- sz/2`` leaving ``(0, 1)``) still return ``1e7`` on the
    optimization path.
    """
    raise NotImplementedError(_DDM7_NOT_SUPPORTED)
    # --- disabled in this release; preserved for future support ---
    # v_coef = float(params[0])
    # a = float(a_xform(params[1]))
    # t0 = float(t0_xform(params[2]))
    # z = float(norm2alpha(params[3]))
    # sv = float(sv_xform(params[4]))
    # st = float(st_xform(params[5]))
    # sz = float(sz_xform(params[6]))
    #
    # infeasible = (
    #     not (-20.0 <= v_coef <= 20.0)
    #     or not (1e-5 <= a <= A_CAP)
    #     or not (0.0 < t0 < T0_CAP)
    #     or not (0.0 < z < 1.0)
    #     or sv < 0.0
    #     or not (0.0 <= st < ST_CAP)
    #     or not (0.0 <= sz < SZ_CAP)
    #     or (z - sz / 2.0) <= 0.0
    #     or (z + sz / 2.0) >= 1.0
    # )
    # if infeasible and output != "all":
    #     return 1e7
    #
    # rt = np.asarray(rt, dtype=float).ravel()
    # choice = np.asarray(choice).ravel().astype(int)
    # value_high = np.asarray(value_high, dtype=float).ravel()
    # value_low = np.asarray(value_low, dtype=float).ravel()
    #
    # v_mean = v_coef * (value_high - value_low)   # drift toward HIGH (upper)
    # nll, logdens = _wfpt_nll(rt, choice, v_mean, a, t0, z, sv, st, sz,
    #                          n_st=n_st, n_sz=n_sz)
    #
    # if output == "all":
    #     return {
    #         "params": [v_coef, a, t0, z, sv, st, sz],
    #         "rt": rt, "choice": choice,
    #         "value_high": value_high, "value_low": value_low,
    #         "v": v_mean, "sv": sv, "st": st, "sz": sz,
    #         "logdens": logdens, "nll": nll,
    #     }
    # return calc_fval(nll, params, prior=prior, output=output)


def ddm7_sim_paths(params, ntrials: int = 25, dt: float = 1e-3,
                   max_time: float = 4.0, rng=None) -> dict:
    """Simulate a SMALL number of trials of the seven-parameter high-vs-low
    VALUE model while recording the full evidence trajectory of every trial, for
    visualization.

    ``params`` is a single subject's natural-space vector
    ``[v_coef, a, t0, z, sv, st, sz]`` (shape ``(7,)`` or ``(1, 7)``). Task
    variables and per-trial draws follow :func:`ddm7_sim` exactly; the default
    ``dt = 1e-3`` is coarser than ``ddm7_sim``'s ``1e-4`` because the purpose is
    drawing legible paths, not unbiased RT distributions. Returns, per trial:
    ``t`` (time arrays offset by ``t0_draw``), ``x`` (evidence arrays), plus
    ``choice`` (1 = HIGH/upper, 0 = LOW/lower), ``crossed``, ``rt``, ``v``
    (mean drift), ``v_draw``, ``value_high``, ``value_low``, ``t0_draw``,
    ``z_draw`` and ``params``.
    """
    raise NotImplementedError(_DDM7_NOT_SUPPORTED)
    # --- disabled in this release; preserved for future support ---
    # params = np.asarray(params, dtype=float).reshape(-1)
    # if params.shape[0] != 7:
    #     raise ValueError(
    #         "params must be a single subject's 7-vector: [v_coef, a, t0, z, sv, st, sz]")
    # v_coef, a, t0, z, sv, st, sz = (float(p) for p in params)
    # _validate_core(np.array([a]), np.array([t0]), np.array([z]), max_time)
    # _validate_variability(np.array([sv]), np.array([st]), np.array([sz]), np.array([z]))
    # if rng is None:
    #     rng = np.random.default_rng()
    #
    # value_high, value_low = _draw_highlow_task(ntrials, rng)
    # v_mean = v_coef * (value_high - value_low)
    # v_draw = v_mean + sv * rng.standard_normal(ntrials)
    # z_draw = z + sz * (rng.random(ntrials) - 0.5)
    # t0_draw = t0 + st * (rng.random(ntrials) - 0.5)
    #
    # ts, xs, choice, crossed, rt = _simulate_paths(
    #     v_draw, a, z_draw, t0_draw, dt, max_time, rng)
    #
    # return {
    #     "params": params.copy(),
    #     "t": ts, "x": xs,
    #     "choice": choice, "crossed": crossed, "rt": rt,
    #     "v": v_mean, "v_draw": v_draw,
    #     "value_high": value_high, "value_low": value_low,
    #     "t0_draw": t0_draw, "z_draw": z_draw,
    # }


ddm4_desc = """Four-parameter drift-diffusion model (DDM) of a HIGH-vs-LOW
value-based choice (deterministic, no gamble). Each trial offers two certain
amounts (value_high > value_low) and the agent should choose the higher one;
choice and response time are modelled by a two-boundary Wiener diffusion
(upper = HIGH/correct, lower = LOW/error; start = z*a) with the Navarro & Fuss
(2009) first-passage likelihood. Drift is the value gap: v = v_coef *
(value_high - value_low) >= 0, so it points at the upper (correct) boundary and
the task has a well-defined accuracy; difficulty = the gap size. Recommended for
parameter recovery.
Free parameters: v_coef (drift scaling), a (boundary separation),
t0 (non-decision time), z (start-point bias)."""
ddm4_id = "ddm4"
ddm4_spec = {
    "task": "high_vs_low_value",
    "ddm": {"drift": ["v_coef"], "boundary": ["a"], "ndt": ["t0"], "bias": ["z"]},
    "likelihood": "navarro_fuss_2009_wfpt",
    "simulation": "vectorized_euler_maruyama(dt=1e-4)",
    "boundaries": {"upper": "high (correct)", "lower": "low (error)"},
}
ddm4_model = ModelSpec(
    id=ddm4_id, spec=ddm4_spec, desc=ddm4_desc.strip(),
    params=None, sim=ddm4_sim, fit=ddm4_fit,
)

ddm7_desc = """Seven-parameter drift-diffusion model (DDM) of a HIGH-vs-LOW
value-based choice, generalizing ddm4 with three across-trial variability
parameters: sv (SD of a Normal on the trial drift), st (full width of a Uniform
on the non-decision time t0) and sz (full width of a Uniform on the RELATIVE
start point z). Each trial offers two certain amounts (value_high > value_low);
the agent should choose the higher (upper = HIGH/correct, lower = LOW/error;
start = z*a). Drift is the value gap: v = v_coef * (value_high - value_low) >= 0.
sv is marginalized analytically (Ratcliff & Tuerlinckx, 2002) on top of the
Navarro & Fuss (2009) density; st and sz by Gauss-Legendre quadrature. Setting
sv = st = sz = 0 reduces the model exactly to ddm4. The variability parameters
are weakly identified for recovery in this design (see the module docstring's
Recoverability note); ddm4 is recommended for recovery.
Free parameters: v_coef (drift scaling), a (boundary separation),
t0 (non-decision time), z (start-point bias), sv (drift SD),
st (non-decision-time width), sz (start-point width)."""
# NOTE: ddm7/ddm7_lotto are disabled in this release (functions raise NotImplementedError).
ddm7_id = "ddm7"
ddm7_spec = {
    "task": "high_vs_low_value",
    "ddm": {"drift": ["v_coef"], "boundary": ["a"], "ndt": ["t0"], "bias": ["z"]},
    "variability": {"drift_sd": ["sv"], "ndt_width": ["st"], "start_width": ["sz"]},
    "likelihood": "navarro_fuss_2009_wfpt + analytic_sv (ratcliff_tuerlinckx_2002)",
    "integration": "gauss_legendre(sz, st)",
    "simulation": "vectorized_euler_maruyama(dt=1e-4)",
    "boundaries": {"upper": "high (correct)", "lower": "low (error)"},
    "reduction_of": "ddm4",
}
ddm7_model = ModelSpec(
    id=ddm7_id, spec=ddm7_spec, desc=ddm7_desc.strip(),
    params=None, sim=ddm7_sim, fit=ddm7_fit,
)


# =============================================================================
# TASK 2 --- Safe-vs-risky GAMBLE choice (a lottery vs a certain amount)
# =============================================================================
def ddm4_lotto_sim(params: np.ndarray, ntrials: int = 150, dt: float = 1e-4,
                   max_time: float = 8.0, rng=None, **kwargs) -> dict:
    """Simulate the **four-parameter** safe-vs-risky GAMBLE task.

    Natural-space columns ``[v_coef, a, t0, z]``, shape ``(nsubjects, 4)``. Each
    trial pits a RISKY gamble (``EV_risky = p_win * payoff``) against a SAFE
    certain amount; drift ``v = v_coef * (EV_risky - safe)`` can point either way.
    The start is ``z * a`` with unit noise. Returns per-trial ``rt``, ``choice``
    (1 = RISKY/upper, 0 = SAFE/lower), ``ev_risky``, ``safe``, ``v`` (mean
    drift), each ``(nsubjects, ntrials)``, plus ``params``.
    """
    params = np.asarray(params, dtype=float)
    if params.ndim != 2 or params.shape[1] != 4:
        raise ValueError("params must have shape (nsubjects, 4): [v_coef, a, t0, z]")
    nsubjects = params.shape[0]
    all_v_coef, all_a, all_t0, all_z = (params[:, i] for i in range(4))
    _validate_core(all_a, all_t0, all_z, max_time)
    if rng is None:
        rng = np.random.default_rng()

    n = nsubjects * ntrials
    v_coef = np.repeat(all_v_coef, ntrials)
    a = np.repeat(all_a, ntrials)
    z = np.repeat(all_z, ntrials)
    t0 = np.repeat(all_t0, ntrials)

    ev_risky, safe = _draw_lotto_task(n, rng)
    v_mean = v_coef * (ev_risky - safe)

    rt, choice, _ = _simulate_diffusion(v_mean, a, z, t0, dt, max_time, rng)

    shape = (nsubjects, ntrials)
    return {
        "params": np.column_stack([all_v_coef, all_a, all_t0, all_z]),
        "rt": rt.reshape(shape),
        "choice": choice.reshape(shape),
        "ev_risky": ev_risky.reshape(shape),
        "safe": safe.reshape(shape),
        "v": v_mean.reshape(shape),
    }


def ddm4_lotto_fit(params, rt, choice, ev_risky, safe, prior=None,
                   output: str = "npl"):
    """EM-compatible objective for the **four-parameter** safe-vs-risky GAMBLE
    task.

    Normalized columns ``[v_coef, a, t0, z]`` decoded as in :func:`ddm4_fit`.
    Drift ``v = v_coef * (EV_risky - safe)``; ``choice`` 1 = RISKY (upper),
    0 = SAFE (lower). With no non-decision-time spread, ``np.any(rt - t0 <= 0)``
    is a hard ``1e7`` penalty on the optimization path; ``output="all"`` still
    returns a well-formed dict.
    """
    v_coef = float(params[0])
    a = float(a_xform(params[1]))
    t0 = float(t0_xform(params[2]))
    z = float(norm2alpha(params[3]))

    rt = np.asarray(rt, dtype=float).ravel()
    choice = np.asarray(choice).ravel().astype(int)
    ev_risky = np.asarray(ev_risky, dtype=float).ravel()
    safe = np.asarray(safe, dtype=float).ravel()

    infeasible = (
        not (-20.0 <= v_coef <= 20.0)
        or not (1e-5 <= a <= A_CAP)
        or not (0.0 < t0 < T0_CAP)
        or not (0.0 < z < 1.0)
        or bool(np.any(rt - t0 <= 0.0))
    )
    if infeasible and output != "all":
        return 1e7

    v_mean = v_coef * (ev_risky - safe)          # per-trial drift
    nll, logdens = _wfpt_nll(rt, choice, v_mean, a, t0, z, 0.0, 0.0, 0.0)

    if output == "all":
        return {
            "params": [v_coef, a, t0, z],
            "rt": rt, "choice": choice,
            "ev_risky": ev_risky, "safe": safe,
            "v": v_mean, "logdens": logdens, "nll": nll,
        }
    return calc_fval(nll, params, prior=prior, output=output)


def ddm7_lotto_sim(params: np.ndarray, ntrials: int = 150, dt: float = 1e-4,
                   max_time: float = 8.0, rng=None, **kwargs) -> dict:
    """Simulate the **seven-parameter** safe-vs-risky GAMBLE task.

    Natural-space columns ``[v_coef, a, t0, z, sv, st, sz]``, shape
    ``(nsubjects, 7)``. Same gamble task as :func:`ddm4_lotto_sim`, but drift,
    non-decision time and relative start are drawn from their across-trial
    distributions ``v_draw ~ Normal(v_mean, sv)``, ``t0_draw ~ U(t0 +/- st/2)``,
    ``z_draw ~ U(z +/- sz/2)`` with ``v_mean = v_coef * (EV_risky - safe)``.
    Returns the same keys as :func:`ddm4_lotto_sim` plus ``sv``, ``st``, ``sz``
    broadcast to ``(nsubjects, ntrials)``.
    """
    raise NotImplementedError(_DDM7_NOT_SUPPORTED)
    # --- disabled in this release; preserved for future support ---
    # params = np.asarray(params, dtype=float)
    # if params.ndim != 2 or params.shape[1] != 7:
    #     raise ValueError(
    #         "params must have shape (nsubjects, 7): [v_coef, a, t0, z, sv, st, sz]")
    # nsubjects = params.shape[0]
    # all_v_coef, all_a, all_t0, all_z, all_sv, all_st, all_sz = (
    #     params[:, i] for i in range(7))
    # _validate_core(all_a, all_t0, all_z, max_time)
    # _validate_variability(all_sv, all_st, all_sz, all_z)
    # if rng is None:
    #     rng = np.random.default_rng()
    #
    # n = nsubjects * ntrials
    # v_coef = np.repeat(all_v_coef, ntrials)
    # a = np.repeat(all_a, ntrials)
    # z = np.repeat(all_z, ntrials)
    # t0 = np.repeat(all_t0, ntrials)
    # sv = np.repeat(all_sv, ntrials)
    # st = np.repeat(all_st, ntrials)
    # sz = np.repeat(all_sz, ntrials)
    #
    # ev_risky, safe = _draw_lotto_task(n, rng)
    # v_mean = v_coef * (ev_risky - safe)
    # v_draw = v_mean + sv * rng.standard_normal(n)
    # z_draw = z + sz * (rng.random(n) - 0.5)
    # t0_draw = t0 + st * (rng.random(n) - 0.5)
    #
    # rt, choice, _ = _simulate_diffusion(v_draw, a, z_draw, t0_draw, dt, max_time, rng)
    #
    # shape = (nsubjects, ntrials)
    # return {
    #     "params": np.column_stack(
    #         [all_v_coef, all_a, all_t0, all_z, all_sv, all_st, all_sz]),
    #     "rt": rt.reshape(shape),
    #     "choice": choice.reshape(shape),
    #     "ev_risky": ev_risky.reshape(shape),
    #     "safe": safe.reshape(shape),
    #     "v": v_mean.reshape(shape),
    #     "sv": sv.reshape(shape).copy(),
    #     "st": st.reshape(shape).copy(),
    #     "sz": sz.reshape(shape).copy(),
    # }


def ddm7_lotto_fit(params, rt, choice, ev_risky, safe, prior=None,
                   output: str = "npl", n_st: int = 11, n_sz: int = 11):
    """EM-compatible objective for the **seven-parameter** safe-vs-risky GAMBLE
    task.

    Normalized columns ``[v_coef, a, t0, z, sv, st, sz]`` decoded as in
    :func:`ddm7_fit`. Drift ``v = v_coef * (EV_risky - safe)``; ``choice`` 1 =
    RISKY (upper), 0 = SAFE (lower). ``t0`` is spread over ``U(t0 +/- st/2)`` so
    infeasible non-decision draws are handled per trial (no hard ``rt <= t0``
    rejection); out-of-bounds parameters (or ``z +/- sz/2`` leaving ``(0, 1)``)
    return ``1e7`` on the optimization path.
    """
    raise NotImplementedError(_DDM7_NOT_SUPPORTED)
    # --- disabled in this release; preserved for future support ---
    # v_coef = float(params[0])
    # a = float(a_xform(params[1]))
    # t0 = float(t0_xform(params[2]))
    # z = float(norm2alpha(params[3]))
    # sv = float(sv_xform(params[4]))
    # st = float(st_xform(params[5]))
    # sz = float(sz_xform(params[6]))
    #
    # infeasible = (
    #     not (-20.0 <= v_coef <= 20.0)
    #     or not (1e-5 <= a <= A_CAP)
    #     or not (0.0 < t0 < T0_CAP)
    #     or not (0.0 < z < 1.0)
    #     or sv < 0.0
    #     or not (0.0 <= st < ST_CAP)
    #     or not (0.0 <= sz < SZ_CAP)
    #     or (z - sz / 2.0) <= 0.0
    #     or (z + sz / 2.0) >= 1.0
    # )
    # if infeasible and output != "all":
    #     return 1e7
    #
    # rt = np.asarray(rt, dtype=float).ravel()
    # choice = np.asarray(choice).ravel().astype(int)
    # ev_risky = np.asarray(ev_risky, dtype=float).ravel()
    # safe = np.asarray(safe, dtype=float).ravel()
    #
    # v_mean = v_coef * (ev_risky - safe)          # per-trial mean drift
    # nll, logdens = _wfpt_nll(rt, choice, v_mean, a, t0, z, sv, st, sz,
    #                          n_st=n_st, n_sz=n_sz)
    #
    # if output == "all":
    #     return {
    #         "params": [v_coef, a, t0, z, sv, st, sz],
    #         "rt": rt, "choice": choice,
    #         "ev_risky": ev_risky, "safe": safe,
    #         "v": v_mean, "sv": sv, "st": st, "sz": sz,
    #         "logdens": logdens, "nll": nll,
    #     }
    # return calc_fval(nll, params, prior=prior, output=output)


def ddm7_lotto_sim_paths(params, ntrials: int = 25, dt: float = 1e-3,
                         max_time: float = 4.0, rng=None) -> dict:
    """Simulate a SMALL number of trials of the seven-parameter safe-vs-risky
    GAMBLE model while recording the full evidence trajectory of every trial, for
    visualization.

    ``params`` is a single subject's natural-space vector
    ``[v_coef, a, t0, z, sv, st, sz]``. Task variables and per-trial draws follow
    :func:`ddm7_lotto_sim` exactly. Returns, per trial: ``t``, ``x``, ``choice``
    (1 = RISKY/upper, 0 = SAFE/lower), ``crossed``, ``rt``, ``v``, ``v_draw``,
    ``ev_risky``, ``safe``, ``t0_draw``, ``z_draw`` and ``params``.
    """
    raise NotImplementedError(_DDM7_NOT_SUPPORTED)
    # --- disabled in this release; preserved for future support ---
    # params = np.asarray(params, dtype=float).reshape(-1)
    # if params.shape[0] != 7:
    #     raise ValueError(
    #         "params must be a single subject's 7-vector: [v_coef, a, t0, z, sv, st, sz]")
    # v_coef, a, t0, z, sv, st, sz = (float(p) for p in params)
    # _validate_core(np.array([a]), np.array([t0]), np.array([z]), max_time)
    # _validate_variability(np.array([sv]), np.array([st]), np.array([sz]), np.array([z]))
    # if rng is None:
    #     rng = np.random.default_rng()
    #
    # ev_risky, safe = _draw_lotto_task(ntrials, rng)
    # v_mean = v_coef * (ev_risky - safe)
    # v_draw = v_mean + sv * rng.standard_normal(ntrials)
    # z_draw = z + sz * (rng.random(ntrials) - 0.5)
    # t0_draw = t0 + st * (rng.random(ntrials) - 0.5)
    #
    # ts, xs, choice, crossed, rt = _simulate_paths(
    #     v_draw, a, z_draw, t0_draw, dt, max_time, rng)
    #
    # return {
    #     "params": params.copy(),
    #     "t": ts, "x": xs,
    #     "choice": choice, "crossed": crossed, "rt": rt,
    #     "v": v_mean, "v_draw": v_draw,
    #     "ev_risky": ev_risky, "safe": safe,
    #     "t0_draw": t0_draw, "z_draw": z_draw,
    # }


ddm4_lotto_desc = """Four-parameter drift-diffusion model (DDM) of a
safe-vs-risky GAMBLE choice. Each trial pits a SAFE certain amount against a
RISKY gamble (EV_risky = p*payoff); choice and response time are modelled by a
two-boundary Wiener diffusion (upper = RISKY, lower = SAFE; start = z*a) with
the Navarro & Fuss (2009) first-passage likelihood. Drift is the risky-minus-safe
value difference: v = v_coef * (EV_risky - safe), which can point toward either
boundary. Recommended for parameter recovery.
Free parameters: v_coef (drift scaling), a (boundary separation),
t0 (non-decision time), z (start-point bias)."""
ddm4_lotto_id = "ddm4_lotto"
ddm4_lotto_spec = {
    "task": "safe_vs_risky_gamble",
    "ddm": {"drift": ["v_coef"], "boundary": ["a"], "ndt": ["t0"], "bias": ["z"]},
    "likelihood": "navarro_fuss_2009_wfpt",
    "simulation": "vectorized_euler_maruyama(dt=1e-4)",
    "boundaries": {"upper": "risky", "lower": "safe"},
}
ddm4_lotto_model = ModelSpec(
    id=ddm4_lotto_id, spec=ddm4_lotto_spec, desc=ddm4_lotto_desc.strip(),
    params=None, sim=ddm4_lotto_sim, fit=ddm4_lotto_fit,
)

ddm7_lotto_desc = """Seven-parameter drift-diffusion model (DDM) of a
safe-vs-risky GAMBLE choice, generalizing ddm4_lotto with three across-trial
variability parameters: sv (SD of a Normal on the trial drift), st (full width
of a Uniform on the non-decision time t0) and sz (full width of a Uniform on the
RELATIVE start point z). Each trial pits a SAFE certain amount against a RISKY
gamble (EV_risky = p*payoff); choice and response time are modelled by a
two-boundary Wiener diffusion (upper = RISKY, lower = SAFE; start = z*a). Drift
is the risky-minus-safe value difference: v = v_coef * (EV_risky - safe). sv is
marginalized analytically (Ratcliff & Tuerlinckx, 2002) on top of the Navarro &
Fuss (2009) density; st and sz by Gauss-Legendre quadrature. Setting
sv = st = sz = 0 reduces the model exactly to ddm4_lotto. The variability
parameters are weakly identified for recovery in this design (see the module
docstring's Recoverability note); ddm4_lotto is recommended for recovery.
Free parameters: v_coef (drift scaling), a (boundary separation),
t0 (non-decision time), z (start-point bias), sv (drift SD),
st (non-decision-time width), sz (start-point width)."""
# NOTE: ddm7/ddm7_lotto are disabled in this release (functions raise NotImplementedError).
ddm7_lotto_id = "ddm7_lotto"
ddm7_lotto_spec = {
    "task": "safe_vs_risky_gamble",
    "ddm": {"drift": ["v_coef"], "boundary": ["a"], "ndt": ["t0"], "bias": ["z"]},
    "variability": {"drift_sd": ["sv"], "ndt_width": ["st"], "start_width": ["sz"]},
    "likelihood": "navarro_fuss_2009_wfpt + analytic_sv (ratcliff_tuerlinckx_2002)",
    "integration": "gauss_legendre(sz, st)",
    "simulation": "vectorized_euler_maruyama(dt=1e-4)",
    "boundaries": {"upper": "risky", "lower": "safe"},
    "reduction_of": "ddm4_lotto",
}
ddm7_lotto_model = ModelSpec(
    id=ddm7_lotto_id, spec=ddm7_lotto_spec, desc=ddm7_lotto_desc.strip(),
    params=None, sim=ddm7_lotto_sim, fit=ddm7_lotto_fit,
)
