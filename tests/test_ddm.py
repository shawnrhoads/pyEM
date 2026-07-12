import numpy as np
import pytest
from scipy.special import logit

from pyem.models.ddm import (
    _wfpt_logf, wfpt_sv_logpdf, wfpt_logpdf, _marginal_logpdf,
    ddm4_sim, ddm4_fit, ddm4_model,
    ddm7_sim, ddm7_fit, ddm7_model, ddm7_sim_paths,
    ddm4_lotto_sim, ddm4_lotto_fit, ddm4_lotto_model,
    ddm7_lotto_sim, ddm7_lotto_fit, ddm7_lotto_model, ddm7_lotto_sim_paths,
    a_xform, t0_xform, sv_xform, st_xform, sz_xform,
    A_CAP, T0_CAP, T0_SHIFT, ST_CAP, SZ_CAP, SV_CAP, _EPS_VAR,
)
from pyem.utils.math import alpha2norm, beta2norm


def _trapz(y, x):
    """numpy-version-agnostic trapezoid (np.trapz removed in numpy 2.x)."""
    return float(np.sum((y[:-1] + y[1:]) / 2.0 * np.diff(x)))


# ----------------------------------------------------------------------------
# 1. sv=0 analytic density reduces to the base WFPT
# ----------------------------------------------------------------------------
def _naive_logdens(rt, v, a, z, nterms=5000):
    """Independent large-time-series reference for the lower-boundary density
    (Navarro & Fuss, 2009, Eq. 4 with the drift factor pulled out)."""
    k = np.arange(1, nterms + 1)
    s = np.sum(k * np.exp(-(k**2) * np.pi**2 * rt / (2 * a**2)) * np.sin(k * np.pi * z))
    return np.log(np.pi / (a**2) * np.exp(-v * a * z - v**2 * rt / 2.0) * s)


@pytest.mark.parametrize(
    "rt,v,a,z",
    [
        (0.5, 1.0, 1.2, 0.5),
        (0.8, -0.5, 1.5, 0.4),
        (0.3, 0.0, 1.0, 0.5),
        (2.5, 0.2, 1.5, 0.45),
        (0.05, 1.0, 2.0, 0.5),   # tt = 0.0125 -> small-time series in _wfpt_logf
    ],
)
def test_wfpt_matches_naive_reference(rt, v, a, z):
    got = float(np.asarray(wfpt_logpdf(rt, v, a, z)).ravel()[0])
    ref = _naive_logdens(rt, v, a, z)
    assert abs(got - ref) < 1e-8, (got, ref)


@pytest.mark.parametrize(
    "rt,v,a,z",
    [
        (0.5, 1.0, 1.2, 0.5),
        (0.15, 2.0, 0.8, 0.3),
        (2.5, 0.2, 1.5, 0.45),
    ],
)
def test_wfpt_sv_zero_matches_naive_reference(rt, v, a, z):
    got = float(np.asarray(wfpt_sv_logpdf(rt, v, a, z, sv=0.0)).ravel()[0])
    ref = _naive_logdens(rt, v, a, z)
    assert abs(got - ref) < 1e-8, (got, ref)


# ----------------------------------------------------------------------------
# 2. Analytic sv marginal == dense numerical integration over the drift
#    (the check the earlier implementation never had; validates the
#    Ratcliff-Tuerlinckx formula at NONZERO sv against integrating the base
#    Navarro-Fuss density)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rt,v,a,z,sv",
    [
        (0.5, 1.0, 1.2, 0.5, 0.8),
        (0.8, -0.5, 1.5, 0.4, 1.5),
        (2.5, 0.2, 1.5, 0.45, 2.0),
        (0.3, 2.0, 0.9, 0.3, 1.0),
        (2.79, 2.81, 0.63, 0.72, 2.45),   # large sv*sqrt(rt) stress case
    ],
)
def test_wfpt_sv_matches_numeric_integration(rt, v, a, z, sv):
    got = float(np.asarray(wfpt_sv_logpdf(rt, v, a, z, sv=sv)).ravel()[0])
    vv = np.linspace(v - 12.0 * sv, v + 12.0 * sv, 40001)
    lp = wfpt_logpdf(np.full_like(vv, rt), vv, a, np.full_like(vv, z))
    dens = np.exp(lp) * np.exp(-0.5 * ((vv - v) / sv) ** 2) / (sv * np.sqrt(2 * np.pi))
    num = np.log(_trapz(dens, vv))
    assert abs(got - num) < 1e-6, (got, num)


# ----------------------------------------------------------------------------
# 3. Fixed-order quadrature ~= a dense reference (at the WIDER ddm widths)
# ----------------------------------------------------------------------------
def test_marginal_matches_dense_reference():
    rng = np.random.default_rng(0)
    n = 25
    # rt floor 0.6 keeps every quadrature node tau <= t0 + st/2 = 0.325 strictly
    # below all rt: no truncation discontinuity in the coarse-vs-dense comparison.
    rt = 0.5 + rng.uniform(0.1, 1.5, size=n)
    v = rng.uniform(-1.5, 1.5, size=n)
    zc = np.full(n, 0.5)
    a, t0, sv, st, sz = 1.2, 0.20, 1.0, 0.25, 0.5
    coarse = _marginal_logpdf(rt, v, a, zc, t0, sv, st, sz, n_st=11, n_sz=11)
    dense = _marginal_logpdf(rt, v, a, zc, t0, sv, st, sz, n_st=101, n_sz=101)
    assert np.all(np.abs(coarse - dense) < 1e-3), np.max(np.abs(coarse - dense))


# ----------------------------------------------------------------------------
# 4. Two-boundary marginal density integrates to 1
# ----------------------------------------------------------------------------
def test_marginal_prob_mass_sums_to_one():
    a, z, t0, sv, st, sz = 1.2, 0.5, 0.20, 1.0, 0.2, 0.4
    v = 0.5
    t = np.linspace(1e-3, 30.0, 40000)
    vv = np.full(t.shape, v)
    lower = np.exp(_marginal_logpdf(t, vv, a, np.full(t.shape, z), t0, sv, st, sz))
    upper = np.exp(_marginal_logpdf(t, -vv, a, np.full(t.shape, 1.0 - z), t0, sv, st, sz))
    total = _trapz(lower, t) + _trapz(upper, t)
    assert abs(total - 1.0) < 5e-3, total


# ----------------------------------------------------------------------------
# 5. THE GATE: zero-variability full model == four-parameter base variant on
#    identical data, for BOTH tasks (high-vs-low value, safe-vs-risky gamble)
# ----------------------------------------------------------------------------
def _p4_normalized(v_coef, a, t0, z):
    return np.array([
        v_coef,                          # identity
        logit(a / A_CAP),                # a_xform inverse
        logit(t0 / T0_CAP) + T0_SHIFT,   # t0_xform inverse (shifted logistic)
        alpha2norm(z),                   # norm2alpha inverse
    ], dtype=float)


@pytest.mark.parametrize(
    "sim4,fit4,fit7,cols",
    [
        (ddm4_sim, ddm4_fit, ddm7_fit, ("value_high", "value_low")),
        (ddm4_lotto_sim, ddm4_lotto_fit, ddm7_lotto_fit, ("ev_risky", "safe")),
    ],
    ids=["highlow", "lotto"],
)
def test_full_model_reduces_to_base_variant(sim4, fit4, fit7, cols):
    raw_sv = float(beta2norm(1e-9, max_val=SV_CAP))
    raw_st = -50.0
    raw_sz = -50.0
    assert float(sv_xform(raw_sv)) < _EPS_VAR
    assert float(st_xform(raw_st)) < _EPS_VAR
    assert float(sz_xform(raw_sz)) < _EPS_VAR

    true4 = np.array([[1.5, 1.2, 0.15, 0.5]])
    sim = sim4(true4, ntrials=80, rng=np.random.default_rng(11))
    rt = sim["rt"][0]
    choice = sim["choice"][0]
    col1 = sim[cols[0]][0]
    col2 = sim[cols[1]][0]

    t0 = min(0.15, 0.9 * rt.min())
    p4 = _p4_normalized(v_coef=1.5, a=1.2, t0=t0, z=0.5)
    p7 = np.concatenate([p4, [raw_sv, raw_st, raw_sz]])

    nll_base = fit4(p4, rt, choice, col1, col2, output="nll")
    nll_full = fit7(p7, rt, choice, col1, col2, output="nll")
    assert np.isfinite(nll_base) and nll_base < 1e7
    assert abs(nll_full - nll_base) < 1e-9, (nll_base, nll_full, abs(nll_full - nll_base))


# ----------------------------------------------------------------------------
# 6. Profile-consistency smoke test (gamble task) — regression test for the
#    boundary-overshoot bug. Data simulated by ddm7_lotto_sim (vectorized,
#    dt=1e-4) must place the NLL grid minimum within one grid step of the true
#    sv/st/sz. A per-trial Euler simulator at dt=1e-3 fails this for sv.
# ----------------------------------------------------------------------------
def _profile_nll(data, name, grid, true_p):
    rt = data["rt"][0]
    choice = data["choice"][0]
    v_mean = data["v"][0]
    is_upper = choice == 1
    v_eff = np.where(is_upper, -v_mean, v_mean)
    out = []
    for g in grid:
        p = dict(true_p)
        p[name] = g
        zc = np.where(is_upper, 1.0 - p["z"], p["z"])
        ld = _marginal_logpdf(rt, v_eff, p["a"], zc, p["t0"],
                              p["sv"], p["st"], p["sz"])
        out.append(-np.sum(np.where(np.isfinite(ld), ld, -1e3)))
    return np.asarray(out)


def test_profile_minima_at_truth():
    true_p = {"v_coef": 1.5, "a": 1.3, "t0": 0.20, "z": 0.5,
              "sv": 1.0, "st": 0.15, "sz": 0.4}
    params = np.array([[true_p["v_coef"], true_p["a"], true_p["t0"], true_p["z"],
                        true_p["sv"], true_p["st"], true_p["sz"]]])
    data = ddm7_lotto_sim(params, ntrials=2000, rng=np.random.default_rng(7))

    grids = {
        "sv": np.linspace(0.0, 2.0, 9),      # step 0.25
        "st": np.linspace(0.05, 0.30, 6),    # step 0.05
        "sz": np.linspace(0.0, 0.8, 9),      # step 0.10
    }
    for name, grid in grids.items():
        nll = _profile_nll(data, name, grid, true_p)
        argmin = grid[int(np.argmin(nll))]
        step = grid[1] - grid[0]
        assert abs(argmin - true_p[name]) <= step + 1e-9, (
            f"{name}: profile argmin {argmin} not within one grid step "
            f"of truth {true_p[name]}"
        )


# ----------------------------------------------------------------------------
# 7. GAMBLE task (ddm4_lotto / ddm7_lotto): fit finite, penalties, sim shapes
# ----------------------------------------------------------------------------
def _p7_normalized(v_coef, a, t0, z, sv, st, sz):
    return np.array([
        v_coef,
        logit(a / A_CAP),
        logit(t0 / T0_CAP) + T0_SHIFT,
        alpha2norm(z),
        beta2norm(sv, max_val=SV_CAP),
        logit(st / ST_CAP),
        logit(sz / SZ_CAP),
    ], dtype=float)


def test_ddm7_lotto_fit_finite_on_valid_data():
    true7 = np.array([[1.5, 1.2, 0.15, 0.5, 1.0, 0.10, 0.20]])
    sim = ddm7_lotto_sim(true7, ntrials=60, rng=np.random.default_rng(3))
    rt = sim["rt"][0]
    t0 = min(0.15, 0.9 * rt.min())
    p7 = _p7_normalized(1.5, 1.2, t0, 0.5, 1.0, 0.10, 0.20)
    val = ddm7_lotto_fit(p7, sim["rt"][0], sim["choice"][0],
                         sim["ev_risky"][0], sim["safe"][0], output="nll")
    assert np.isfinite(val) and val < 1e7

    out = ddm7_lotto_fit(p7, sim["rt"][0], sim["choice"][0],
                         sim["ev_risky"][0], sim["safe"][0], output="all")
    assert set(["params", "nll", "sv", "st", "sz", "logdens"]).issubset(out.keys())
    assert np.isfinite(out["nll"])


def test_ddm7_lotto_fit_penalty_on_sz_out_of_bounds():
    rng = np.random.default_rng(2)
    n = 30
    ev_risky = rng.uniform(0.5, 3.0, size=n)
    safe = ev_risky - rng.uniform(-1.0, 1.0, size=n)
    choice = rng.integers(0, 2, size=n)
    rt = 0.2 + rng.uniform(0.2, 1.0, size=n)
    # z ~ 0.95, sz ~ 0.85 -> z + sz/2 >= 1 -> infeasible
    p7 = _p7_normalized(1.0, 1.2, 0.15, 0.95, 0.3, 0.05, 0.85)
    assert ddm7_lotto_fit(p7, rt, choice, ev_risky, safe, output="nll") == 1e7


def test_ddm7_lotto_sim_shapes():
    true7 = np.array([
        [1.5, 1.2, 0.15, 0.5, 0.8, 0.10, 0.20],
        [2.0, 1.6, 0.20, 0.45, 1.2, 0.15, 0.30],
    ])
    sim = ddm7_lotto_sim(true7, ntrials=50, rng=np.random.default_rng(0))
    assert sim["rt"].shape == (2, 50)
    assert sim["choice"].shape == (2, 50)
    assert sim["ev_risky"].shape == (2, 50)
    assert sim["safe"].shape == (2, 50)
    assert sim["params"].shape == (2, 7)
    assert sim["sv"].shape == (2, 50)
    assert set(np.unique(sim["choice"])).issubset({0, 1})
    assert (sim["rt"] > 0).all()


def test_ddm7_lotto_sim_rejects_bad_params():
    with pytest.raises(ValueError):
        ddm7_lotto_sim(np.array([[1.5, 1.2, 0.15, 0.5]]))                    # wrong ncols
    with pytest.raises(ValueError):
        ddm7_lotto_sim(np.array([[1.5, -1.0, 0.15, 0.5, 0.5, 0.1, 0.2]]))    # a <= 0
    with pytest.raises(ValueError):
        ddm7_lotto_sim(np.array([[1.5, 1.2, 0.15, 0.9, 0.5, 0.1, 0.5]]))     # z+sz/2 >= 1


def test_variabilities_change_the_density():
    rt = np.array([0.5]); v = np.array([0.8]); a, z, t0 = 1.2, 0.5, 0.15
    base_lp = float(_marginal_logpdf(rt, v, a, z, t0, 0.0, 0.0, 0.0)[0])
    sv_lp = float(_marginal_logpdf(rt, v, a, z, t0, 0.6, 0.0, 0.0)[0])
    st_lp = float(_marginal_logpdf(rt, v, a, z, t0, 0.0, 0.2, 0.0)[0])
    sz_lp = float(_marginal_logpdf(rt, v, a, z, t0, 0.0, 0.0, 0.3)[0])
    assert abs(sv_lp - base_lp) > 1e-3, ("sv had no effect", sv_lp, base_lp)
    assert abs(st_lp - base_lp) > 1e-3, ("st had no effect", st_lp, base_lp)
    assert abs(sz_lp - base_lp) > 1e-3, ("sz had no effect", sz_lp, base_lp)


def test_ddm4_lotto_sim_shapes_and_validation():
    true4 = np.array([
        [1.5, 1.2, 0.15, 0.5],
        [2.0, 1.6, 0.20, 0.45],
    ])
    sim = ddm4_lotto_sim(true4, ntrials=50, rng=np.random.default_rng(0))
    assert sim["params"].shape == (2, 4)
    assert sim["rt"].shape == (2, 50)
    assert sim["choice"].shape == (2, 50)
    assert set(np.unique(sim["choice"])).issubset({0, 1})
    assert (sim["rt"] > 0).all()
    with pytest.raises(ValueError):
        ddm4_lotto_sim(np.array([[1.5, 1.2, 0.15, 0.5, 0.4, 0.1, 0.2]]))     # 7 cols rejected


def test_ddm4_lotto_fit_finite_and_t0_penalty():
    sim = ddm4_lotto_sim(np.array([[1.5, 1.2, 0.15, 0.5]]), ntrials=60,
                         rng=np.random.default_rng(3))
    rt = sim["rt"][0]
    t0 = min(0.15, 0.9 * rt.min())
    p4 = _p4_normalized(1.5, 1.2, t0, 0.5)
    val = ddm4_lotto_fit(p4, sim["rt"][0], sim["choice"][0],
                         sim["ev_risky"][0], sim["safe"][0], output="nll")
    assert np.isfinite(val) and val < 1e7

    out = ddm4_lotto_fit(p4, sim["rt"][0], sim["choice"][0],
                         sim["ev_risky"][0], sim["safe"][0], output="all")
    assert set(["params", "nll", "logdens", "v"]).issubset(out.keys())
    assert np.isfinite(out["nll"])

    # a non-decision time above the fastest RT is a hard 1e7 on the optimization path
    t0_bad = 1.05 * rt.min()
    assert t0_bad < T0_CAP, "seed produced an unexpectedly slow fastest RT"
    p4_bad = _p4_normalized(1.5, 1.2, t0_bad, 0.5)
    assert ddm4_lotto_fit(p4_bad, sim["rt"][0], sim["choice"][0],
                          sim["ev_risky"][0], sim["safe"][0], output="nll") == 1e7


# ----------------------------------------------------------------------------
# 8. HIGH-vs-LOW value task (ddm4 / ddm7): the new deterministic task
# ----------------------------------------------------------------------------
def test_ddm4_highlow_sim_structure():
    sim = ddm4_sim(np.array([[1.5, 1.3, 0.20, 0.5]]),
                   ntrials=4000, rng=np.random.default_rng(0))
    assert set(["params", "rt", "choice", "value_high", "value_low", "v"]).issubset(sim.keys())
    assert sim["params"].shape == (1, 4)
    assert (sim["value_high"] > sim["value_low"]).all()   # HIGH strictly larger
    assert (sim["v"] >= 0.0).all()                         # drift toward HIGH (upper)
    assert set(np.unique(sim["choice"])).issubset({0, 1})
    assert (sim["rt"] > 0).all()
    # positive v_coef -> drift toward the HIGH (upper) boundary -> mostly correct
    assert sim["choice"].mean() > 0.6, sim["choice"].mean()


def test_ddm4_highlow_fit_finite_and_t0_penalty():
    sim = ddm4_sim(np.array([[1.5, 1.3, 0.20, 0.5]]), ntrials=60,
                   rng=np.random.default_rng(3))
    rt = sim["rt"][0]
    t0 = min(0.20, 0.9 * rt.min())
    p4 = _p4_normalized(1.5, 1.3, t0, 0.5)
    val = ddm4_fit(p4, sim["rt"][0], sim["choice"][0],
                   sim["value_high"][0], sim["value_low"][0], output="nll")
    assert np.isfinite(val) and val < 1e7

    out = ddm4_fit(p4, sim["rt"][0], sim["choice"][0],
                   sim["value_high"][0], sim["value_low"][0], output="all")
    assert set(["params", "nll", "logdens", "v", "value_high", "value_low"]).issubset(out.keys())
    assert np.isfinite(out["nll"])

    t0_bad = 1.05 * rt.min()
    assert t0_bad < T0_CAP, "seed produced an unexpectedly slow fastest RT"
    p4_bad = _p4_normalized(1.5, 1.3, t0_bad, 0.5)
    assert ddm4_fit(p4_bad, sim["rt"][0], sim["choice"][0],
                    sim["value_high"][0], sim["value_low"][0], output="nll") == 1e7


def test_ddm7_highlow_sim_shapes_and_rejects():
    true7 = np.array([
        [1.5, 1.2, 0.15, 0.5, 0.8, 0.10, 0.20],
        [2.0, 1.6, 0.20, 0.45, 1.2, 0.15, 0.30],
    ])
    sim = ddm7_sim(true7, ntrials=50, rng=np.random.default_rng(0))
    assert sim["params"].shape == (2, 7)
    assert sim["rt"].shape == (2, 50)
    assert sim["value_high"].shape == (2, 50)
    assert sim["sv"].shape == (2, 50)
    assert (sim["value_high"] > sim["value_low"]).all()
    assert set(np.unique(sim["choice"])).issubset({0, 1})
    with pytest.raises(ValueError):
        ddm7_sim(np.array([[1.5, 1.2, 0.15, 0.5]]))                        # 4 cols rejected
    with pytest.raises(ValueError):
        ddm7_sim(np.array([[1.5, 1.2, 0.15, 0.9, 0.5, 0.1, 0.5]]))         # z+sz/2 >= 1


def test_ddm7_highlow_reduces_and_fit_all_keys():
    # ddm7 fit output="all" carries the high/low task columns
    sim = ddm7_sim(np.array([[1.5, 1.2, 0.15, 0.5, 1.0, 0.10, 0.20]]), ntrials=60,
                   rng=np.random.default_rng(4))
    p7 = _p7_normalized(1.5, 1.2, min(0.15, 0.9 * sim["rt"][0].min()), 0.5, 1.0, 0.10, 0.20)
    out = ddm7_fit(p7, sim["rt"][0], sim["choice"][0],
                   sim["value_high"][0], sim["value_low"][0], output="all")
    assert set(["params", "nll", "sv", "st", "sz", "logdens",
                "value_high", "value_low"]).issubset(out.keys())
    assert np.isfinite(out["nll"])


# ----------------------------------------------------------------------------
# 9. Trajectory-recording visualization helpers (both tasks)
# ----------------------------------------------------------------------------
def _check_paths(out, a, choice_ok=(0, 1), n=15):
    assert len(out["t"]) == len(out["x"]) == n
    assert out["choice"].shape == out["rt"].shape == (n,)
    for t, x, ch, cr, rt, t0d, zd in zip(out["t"], out["x"], out["choice"],
                                         out["crossed"], out["rt"],
                                         out["t0_draw"], out["z_draw"]):
        assert t.shape == x.shape
        assert np.isclose(t[0], t0d)          # path starts at the trial's ndt
        assert np.isclose(t[-1], rt)          # rt is the last recorded time point
        assert np.isclose(x[0], zd * a)       # path starts at the drawn start point
        assert (x[:-1] > 0.0).all() and (x[:-1] < a).all()   # interior strictly inside
        if cr:
            assert x[-1] in (0.0, a)
            assert ch == (1 if x[-1] == a else 0)


def test_ddm7_sim_paths_highlow():
    p = np.array([1.5, 1.3, 0.25, 0.5, 1.0, 0.10, 0.20])
    out = ddm7_sim_paths(p, ntrials=15, dt=1e-3, rng=np.random.default_rng(5))
    _check_paths(out, a=p[1])
    assert set(["value_high", "value_low", "v_draw"]).issubset(out.keys())
    out2 = ddm7_sim_paths(p, ntrials=15, dt=1e-3, rng=np.random.default_rng(5))
    assert np.allclose(out["rt"], out2["rt"])
    assert all(np.allclose(x1, x2) for x1, x2 in zip(out["x"], out2["x"]))


def test_ddm7_lotto_sim_paths_consistency():
    p = np.array([1.5, 1.3, 0.25, 0.5, 1.0, 0.10, 0.20])
    out = ddm7_lotto_sim_paths(p, ntrials=15, dt=1e-3, rng=np.random.default_rng(5))
    _check_paths(out, a=p[1])
    assert set(["ev_risky", "safe", "v_draw"]).issubset(out.keys())
    out2 = ddm7_lotto_sim_paths(p, ntrials=15, dt=1e-3, rng=np.random.default_rng(5))
    assert np.allclose(out["rt"], out2["rt"])
    assert all(np.allclose(x1, x2) for x1, x2 in zip(out["x"], out2["x"]))


def test_sim_paths_reject_bad_params():
    for paths_fn in (ddm7_sim_paths, ddm7_lotto_sim_paths):
        with pytest.raises(ValueError):
            paths_fn(np.array([1.5, 1.3, 0.25, 0.5]))                       # wrong length
        with pytest.raises(ValueError):
            paths_fn(np.array([1.5, 1.3, 0.25, 0.9, 0.0, 0.0, 0.5]))        # z+sz/2 >= 1

