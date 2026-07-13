import numpy as np
from pyem.api import EMModel
from pyem.models.rl_mf import rw1a1b_sim, rw1a1b_fit
from test_helpers import _simulate_rw_params


def _rw_data(nsubjects=6, nblocks=1, ntrials=6):
    params = _simulate_rw_params(nsubjects)
    sim = rw1a1b_sim(params, nblocks=nblocks, ntrials=ntrials)
    return [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]


def test_keyword_only_prior_fit_func_runs():
    def custom_fit(params, choices, rewards, *, prior=None, output="npl"):
        return rw1a1b_fit(params, choices, rewards, prior=prior, output=output)
    all_data = _rw_data()
    model = EMModel(all_data=all_data, fit_func=custom_fit, param_names=["beta", "alpha"])
    res = model.fit(verbose=0, seed=0)
    assert np.all(np.isfinite(res.m))


def test_seed_none_is_stochastic():
    all_data = _rw_data()
    m1 = EMModel(all_data, rw1a1b_fit, ["beta", "alpha"]).fit(verbose=0, seed=None).m
    m2 = EMModel(all_data, rw1a1b_fit, ["beta", "alpha"]).fit(verbose=0, seed=None).m
    assert not np.allclose(m1, m2)


def test_seed_fixed_is_reproducible():
    all_data = _rw_data()
    m1 = EMModel(all_data, rw1a1b_fit, ["beta", "alpha"]).fit(verbose=0, seed=0).m
    m2 = EMModel(all_data, rw1a1b_fit, ["beta", "alpha"]).fit(verbose=0, seed=0).m
    assert np.allclose(m1, m2)


def test_scipy_minimize_seed_reproducible():
    all_data = _rw_data()
    m1 = EMModel(all_data, rw1a1b_fit, ["beta", "alpha"]).scipy_minimize(seed=0)["m"]
    m2 = EMModel(all_data, rw1a1b_fit, ["beta", "alpha"]).scipy_minimize(seed=0)["m"]
    assert np.allclose(m1, m2)


def test_dead_kwargs_removed():
    import inspect
    from pyem.api import EMModel
    sig = inspect.signature(EMModel.fit)
    assert "estep_maxit" not in sig.parameters
    assert "convergence_type" not in sig.parameters


def test_compute_lme_method_still_works():
    import numpy as np
    from pyem.api import EMModel
    from pyem.models.rl_mf import rw1a1b_fit, rw1a1b_sim
    from test_helpers import _simulate_rw_params
    params = _simulate_rw_params(6)
    sim = rw1a1b_sim(params, nblocks=1, ntrials=6)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
    model = EMModel(all_data, rw1a1b_fit, ["beta", "alpha"])
    model.fit(verbose=0, seed=0)
    lap, lme, good = model.compute_lme()
    assert np.isfinite(lme)
