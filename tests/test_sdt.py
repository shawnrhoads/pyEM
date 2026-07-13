import numpy as np
from scipy.stats import norm
from pyem.models.sdt import sdt_fit, sdt_sim


def test_sdt_fit_contract_and_value():
    is_old = np.array([1, 0]); resp_old = np.array([1, 1])  # 1 hit, 1 false alarm
    # choose params in NORMALIZED space that decode (via norm2beta / identity) to d'=1, c=0
    from pyem.utils.math import beta2norm
    params = np.array([beta2norm(1.0), 0.0])
    val = sdt_fit(params, is_old, resp_old, output="nll")
    dprime, c = 1.0, 0.0
    p_hit = norm.cdf(dprime/2 - c); p_fa = norm.cdf(-dprime/2 - c)
    expected = -(np.log(p_hit) + np.log(p_fa))
    assert np.isclose(val, expected, atol=1e-6)


def test_sdt_fit_oob_returns_penalty():
    # norm2beta underflows to 0.0 for very negative inputs (< the 1e-5 dprime floor),
    # so the out-of-bounds guard IS reachable and must return the 1e7 penalty.
    is_old = np.array([1, 0, 1, 0]); resp_old = np.array([1, 0, 1, 0])
    assert sdt_fit(np.array([-1e10, 0.0]), is_old, resp_old, output="nll") == 1e7
    # and a normal call still returns a finite contract-compliant dict
    out = sdt_fit(np.array([0.0, 0.0]), is_old, resp_old, output="all")
    assert "params" in out and "nll" in out and np.isfinite(out["nll"])


def test_sdt_fit_nonzero_criterion_value():
    # response-bias (criterion != 0) case, recomputed independently
    from pyem.utils.math import beta2norm
    is_old = np.array([1, 0, 1, 0]); resp_old = np.array([1, 0, 0, 1])
    dprime, c = 1.5, 0.4
    params = np.array([beta2norm(dprime), c])
    val = sdt_fit(params, is_old, resp_old, output="nll")
    p_hit = norm.cdf(dprime / 2 - c); p_fa = norm.cdf(-dprime / 2 - c)
    # trials: (old,resp=1)->hit, (new,resp=0)->correct-rej, (old,resp=0)->miss, (new,resp=1)->fa
    expected = -(np.log(p_hit) + np.log(1 - p_fa) + np.log(1 - p_hit) + np.log(p_fa))
    assert np.isclose(val, expected, atol=1e-6)


def test_sdt_sim_shapes():
    rng_params = np.array([[1.0, 0.0], [2.0, 0.5]])  # natural [dprime, criterion]
    sim = sdt_sim(rng_params, ntrials=50)
    assert sim["is_old"].shape == (2, 50) and sim["resp_old"].shape == (2, 50)


def test_sdt_sim_seed_reproducible():
    import numpy as np
    from pyem.models.sdt import sdt_sim
    true = np.array([[1.5, 0.0]])
    a = sdt_sim(true, ntrials=40, seed=2)
    b = sdt_sim(true, ntrials=40, seed=2)
    assert np.array_equal(a["resp_old"], b["resp_old"])
    c = sdt_sim(true, ntrials=40, seed=3)
    assert not np.array_equal(a["resp_old"], c["resp_old"])
