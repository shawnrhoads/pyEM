import numpy as np
from pyem.models.pt import pt_weight, pt_value, pt_fit, pt_sim


def test_weighting_known_point():
    g = 0.7; p = 0.5
    expected = p ** g / (p ** g + (1 - p) ** g) ** (1 / g)
    assert np.isclose(pt_weight(p, g), expected)


def test_value_gain_loss():
    assert np.isclose(pt_value(4.0, alpha=0.5, beta=0.5, lam=2.0), 2.0)      # 4**0.5
    assert np.isclose(pt_value(-4.0, alpha=0.5, beta=0.5, lam=2.0), -4.0)    # -2*4**0.5


def test_pt_fit_contract():
    rng = np.random.default_rng(0)
    n = 40
    gamble = rng.uniform(-10, 10, (n, 2)); probs = np.full((n, 2), 0.5)
    certain = rng.uniform(-5, 5, n); choice = rng.integers(0, 2, n)
    good = np.zeros(5)
    assert np.isfinite(pt_fit(good, gamble, probs, certain, choice))
    out = pt_fit(good, gamble, probs, certain, choice, output="all")
    assert "params" in out and "nll" in out


def test_pt_sim_shapes():
    params = np.array([[0.8, 0.8, 2.0, 0.6, 1.0], [0.6, 0.7, 1.5, 0.7, 2.0]])
    sim = pt_sim(params, ntrials=30)
    assert sim["gamble"].shape == (2, 30, 2) and sim["choice"].shape == (2, 30)


def test_pt_fit_oob_returns_penalty():
    # norm2beta underflows below the 1e-5 lambda floor for very negative inputs,
    # so the out-of-bounds guard IS reachable and must return the 1e7 penalty.
    n = 4
    gamble = np.zeros((n, 2)); probs = np.full((n, 2), 0.5)
    certain = np.zeros(n); choice = np.zeros(n, dtype=int)
    params = np.array([0.0, 0.0, -1e10, 0.0, 0.0])   # lambda slot -> ~0 (< 1e-5)
    assert pt_fit(params, gamble, probs, certain, choice, output="nll") == 1e7


def test_pt_fit_mixed_gamble_nll_value():
    # Independently recompute the composed weighting+value NLL for a mixed
    # (gain + loss) gamble and compare to pt_fit (not just the primitives in isolation).
    from pyem.utils.math import alpha2norm, beta2norm
    from scipy.special import expit
    alpha, beta, lam, gamma, mu = 0.8, 0.7, 2.0, 0.6, 1.5
    gamble = np.array([[10.0, -5.0]]); probs = np.array([[0.6, 0.4]])
    certain = np.array([1.0]); choice = np.array([1])   # chose the gamble
    params = np.array([alpha2norm(alpha), alpha2norm(beta), beta2norm(lam),
                       alpha2norm(gamma), beta2norm(mu)])

    def w(p, g):
        return p ** g / (p ** g + (1 - p) ** g) ** (1 / g)

    def v(x):
        return x ** alpha if x >= 0 else -lam * (-x) ** beta

    v_g = w(0.6, gamma) * v(10.0) + w(0.4, gamma) * v(-5.0)
    v_c = v(1.0)
    p_gamble = expit(mu * (v_g - v_c))
    expected_nll = -np.log(p_gamble)   # choice == 1
    val = pt_fit(params, gamble, probs, certain, choice, output="nll")
    assert np.isclose(val, expected_nll, atol=1e-6)


def test_pt_sim_seed_reproducible():
    import numpy as np
    from pyem.models.pt import pt_sim
    true = np.array([[0.8, 0.8, 1.5, 0.7, 2.0]])
    a = pt_sim(true, ntrials=30, seed=4)
    b = pt_sim(true, ntrials=30, seed=4)
    assert np.array_equal(a["choice"], b["choice"])
    c = pt_sim(true, ntrials=30, seed=5)
    assert not np.array_equal(a["choice"], c["choice"])
