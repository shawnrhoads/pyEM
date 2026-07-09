import numpy as np
import pytest

from pyem.models.ddm import (
    wfpt_logpdf, ddm_fit, ddm_sim, t0_xform, T0_CAP, T0_SHIFT, A_CAP,
)
from pyem.utils.math import alpha2norm


def _numerical_logdens(rt, v, a, z, nterms=5000):
    """Independent large-time-series reference for the DDM lower-boundary density."""
    k = np.arange(1, nterms + 1)
    s = np.sum(k * np.exp(-(k**2) * np.pi**2 * rt / (2 * a**2)) * np.sin(k * np.pi * z))
    dens = np.pi / (a**2) * np.exp(-v * a * z - v**2 * rt / 2.0) * s
    return np.log(dens)


# ----------------------------------------------------------------------------
# 1. WFPT matches a fine-grid numerical reference
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rt,v,a,z",
    [
        (0.5, 1.0, 1.2, 0.5),
        (0.8, -0.5, 1.5, 0.4),
        (0.3, 0.0, 1.0, 0.5),      # zero drift
        (0.15, 2.0, 0.8, 0.3),     # small-time regime
        (2.5, 0.2, 1.5, 0.45),     # large-time regime
    ],
)
def test_wfpt_matches_numerical_reference(rt, v, a, z):
    got = wfpt_logpdf(rt, v, a, z)
    ref = _numerical_logdens(rt, v, a, z)
    assert abs(got - ref) < 1e-3


def test_wfpt_vectorized_matches_scalar():
    rt = np.array([0.5, 0.8, 0.15, 2.5])
    v = np.array([1.0, -0.5, 2.0, 0.2])
    z = np.array([0.5, 0.4, 0.3, 0.45])
    a = 1.2
    vec = wfpt_logpdf(rt, v, a, z)
    scal = np.array([wfpt_logpdf(float(rt[i]), float(v[i]), a, float(z[i]))
                     for i in range(rt.size)])
    assert np.allclose(vec, scal, atol=1e-12)


def test_wfpt_nonpositive_time_is_neg_inf():
    assert wfpt_logpdf(0.0, 1.0, 1.2, 0.5) == -np.inf
    assert wfpt_logpdf(-0.3, 1.0, 1.2, 0.5) == -np.inf


def test_t0_xform_range():
    xs = np.array([-50.0, -1.0, 0.0, 1.0, 50.0])
    ts = t0_xform(xs)
    # (0, T0_CAP]; expit saturates to exactly 1.0 for large x, giving t0 == T0_CAP
    assert np.all(ts > 0.0) and np.all(ts <= T0_CAP)
    # shifted logistic: raw x=0 maps to a small, always-feasible t0 (~0.05 s)
    assert np.isclose(t0_xform(0.0), T0_CAP * (1.0 / (1.0 + np.exp(T0_SHIFT))))
    assert t0_xform(0.0) < 0.1


# ----------------------------------------------------------------------------
# 2. ddm_fit contract
# ----------------------------------------------------------------------------
def _good_normalized_params(v_coef=1.0, a=1.5, t0=0.15, z=0.5):
    from scipy.special import logit
    return np.array([
        v_coef,                       # identity
        logit(a / A_CAP),             # a_xform inverse (norm2beta cap A_CAP)
        logit(t0 / T0_CAP) + T0_SHIFT,  # t0_xform inverse (shifted logistic)
        alpha2norm(z),                # norm2alpha inverse
    ])


def test_ddm_fit_finite_on_good_params():
    rng = np.random.default_rng(0)
    n = 40
    ev_risky = rng.uniform(0.5, 3.0, size=n)
    safe = ev_risky - rng.uniform(-1.0, 1.0, size=n)
    choice = rng.integers(0, 2, size=n)
    rt = 0.15 + rng.uniform(0.2, 1.0, size=n)   # all > t0=0.15
    params = _good_normalized_params()
    val = ddm_fit(params, rt, choice, ev_risky, safe, output="nll")
    assert np.isfinite(val) and val < 1e7

    out = ddm_fit(params, rt, choice, ev_risky, safe, output="all")
    assert set(["params", "nll", "logdens"]).issubset(out.keys())
    assert np.isfinite(out["nll"])


def test_ddm_fit_rejects_t0_above_all_rts():
    rng = np.random.default_rng(1)
    n = 30
    ev_risky = rng.uniform(0.5, 3.0, size=n)
    safe = ev_risky - rng.uniform(-1.0, 1.0, size=n)
    choice = rng.integers(0, 2, size=n)
    rt = 0.05 + rng.uniform(0.0, 0.02, size=n)   # all tiny RTs
    # force t0 near the cap (~0.5), which exceeds every rt -> penalty
    params = _good_normalized_params(t0=0.49)
    assert ddm_fit(params, rt, choice, ev_risky, safe, output="nll") == 1e7


# ----------------------------------------------------------------------------
# 3. Round-trip: simulate -> fit -> finite estimates
# ----------------------------------------------------------------------------
def test_ddm_sim_shapes_and_roundtrip():
    true = np.array([
        [1.5, 1.2, 0.15, 0.5],
        [2.0, 1.6, 0.20, 0.45],
        [1.0, 1.0, 0.18, 0.55],
    ])
    sim = ddm_sim(true, ntrials=60)
    assert sim["rt"].shape == (3, 60)
    assert sim["choice"].shape == (3, 60)
    assert set(np.unique(sim["choice"])).issubset({0, 1})
    assert (sim["rt"] > 0).all()

    # fit each simulated subject at its (approx) true params -> finite NLL
    for s in range(3):
        params = _good_normalized_params(
            v_coef=true[s, 0], a=true[s, 1], t0=min(true[s, 2], 0.9 * sim["rt"][s].min()),
            z=true[s, 3],
        )
        val = ddm_fit(params, sim["rt"][s], sim["choice"][s],
                      sim["ev_risky"][s], sim["safe"][s], output="nll")
        assert np.isfinite(val) and val < 1e7


def test_wfpt_probability_mass_sums_to_one():
    # The lower-boundary density plus the upper-boundary density (via the
    # reflection v->-v, z->1-z) must integrate to 1 over decision time.
    def _trapz(y, x):  # numpy-version-agnostic (np.trapz removed in numpy 2.x)
        return float(np.sum((y[:-1] + y[1:]) / 2.0 * np.diff(x)))

    t = np.linspace(1e-3, 60.0, 60000)
    for v, a, z in [(1.0, 1.2, 0.5), (-0.5, 1.5, 0.4), (0.8, 2.0, 0.6)]:
        lower = np.exp(np.array([wfpt_logpdf(ti, v, a, z) for ti in t]))
        upper = np.exp(np.array([wfpt_logpdf(ti, -v, a, 1.0 - z) for ti in t]))
        total = _trapz(lower, t) + _trapz(upper, t)
        assert abs(total - 1.0) < 1e-3, (v, a, z, total)
