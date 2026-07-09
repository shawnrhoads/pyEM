import numpy as np
from pyem.api import EMModel
from pyem.models.rl import (
    rw1a1b_sim, rw1a1b_fit,
    rw2a1b_sim, rw2a1b_fit,
)
from pyem.models.bayes import bayes_sim, bayes_fit
from pyem.models.glm import (
    glm_sim, glm_fit, glm_decay_sim, glm_decay_fit,
    logit_sim, logit_fit, glm_ar_sim, glm_ar_fit,
)
from test_helpers import _simulate_rw_params

def test_rw1a1b_fit():
    nsubjects, nblocks, ntrials = 10, 2, 12
    params = _simulate_rw_params(nsubjects)
    sim = rw1a1b_sim(params, nblocks=nblocks, ntrials=ntrials)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
    model = EMModel(all_data=all_data, fit_func=rw1a1b_fit, param_names=["beta", "alpha"])
    res = model.fit(mstep_maxit=5, verbose=0, njobs=1)
    assert res.m.shape == (2, nsubjects)
    assert res.NPL.shape == (nsubjects,)

def test_rw2a1b_fit():
    nsubjects, nblocks, ntrials = 10, 2, 12
    params = _simulate_rw_params(nsubjects, 3)
    sim = rw2a1b_sim(params, nblocks=nblocks, ntrials=ntrials)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
    model = EMModel(
        all_data=all_data,
        fit_func=rw2a1b_fit,
        param_names=["beta", "alpha_pos", "alpha_neg"],
    )
    res = model.fit(mstep_maxit=5, verbose=0, njobs=1)
    assert res.m.shape == (3, nsubjects)
    assert res.NPL.shape == (nsubjects,)

def test_bayes_fit():
    nsubjects, nblocks, ntrials = 10, 2, 10
    true_lambda = np.random.uniform(0.2, 0.8, size=(nsubjects, 1))
    sim = bayes_sim(true_lambda, nblocks=nblocks, ntrials=ntrials)
    all_data = [[sim["choices"][i], sim["observations"][i]] for i in range(nsubjects)]
    model = EMModel(all_data=all_data, fit_func=bayes_fit, param_names=["lambda"])
    res = model.fit(mstep_maxit=5, verbose=0, njobs=1)
    assert res.m.shape == (1, nsubjects)
    assert res.NPL.shape == (nsubjects,)

def test_glm_fit():
    nsubjects, nparams, ntrials = 10, 3, 50
    true_params = np.random.randn(nsubjects, nparams)
    X, Y = glm_sim(true_params, ntrials=ntrials)
    all_data = [[X[i], Y[i]] for i in range(nsubjects)]
    model = EMModel(all_data=all_data, fit_func=glm_fit, param_names=[f"b{i}" for i in range(nparams)])
    res = model.fit(mstep_maxit=5, verbose=0, njobs=1)
    assert res.m.shape == (nparams, nsubjects)
    assert res.NPL.shape == (nsubjects,)

def test_glm_decay_fit():
    nsubjects, nparams, ntrials = 10, 2, 50
    true_params = np.random.randn(nsubjects, nparams + 1)  # all random normal
    true_params[:, -1] = np.random.uniform(0, 1, size=nsubjects)  # overwrite last param with uniform(0,1)
    X, Y = glm_decay_sim(true_params, ntrials=ntrials)
    all_data = [[X[i], Y[i]] for i in range(nsubjects)]
    param_names = [f"b{i}" for i in range(nparams)] + ["gamma"]
    model = EMModel(all_data=all_data, fit_func=glm_decay_fit, param_names=param_names)
    res = model.fit(mstep_maxit=5, verbose=0, njobs=1)
    assert res.m.shape == (nparams + 1, nsubjects)
    assert res.NPL.shape == (nsubjects,)

def test_logit_fit():
    nsubjects, nparams, ntrials = 10, 3, 50
    rng = np.random.default_rng(0)
    true_params = rng.normal(size=(nsubjects, nparams))
    X, Y = logit_sim(true_params, ntrials=ntrials)
    all_data = [[X[i], Y[i]] for i in range(nsubjects)]
    model = EMModel(all_data=all_data, fit_func=logit_fit, param_names=[f"b{i}" for i in range(nparams)])
    res = model.fit(mstep_maxit=5, verbose=0, njobs=1)
    assert res.m.shape == (nparams, nsubjects)
    assert res.NPL.shape == (nsubjects,)

def test_gaussian_mstep_matches_default():
    params = _simulate_rw_params(8)
    sim = rw1a1b_sim(params, nblocks=2, ntrials=8)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
    m_default = EMModel(all_data, rw1a1b_fit, ["beta", "alpha"]).fit(verbose=0, seed=0, mstep_maxit=15, njobs=1).m
    m_explicit = EMModel(all_data, rw1a1b_fit, ["beta", "alpha"]).fit(verbose=0, seed=0, mstep="gaussian", mstep_maxit=15, njobs=1).m
    assert np.array_equal(m_default, m_explicit)


def test_all_msteps_run_finite():
    params = _simulate_rw_params(10)
    sim = rw1a1b_sim(params, nblocks=2, ntrials=8)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
    for mstep in ("gaussian", "laplace", "student_t", "cauchy"):
        res = EMModel(all_data, rw1a1b_fit, ["beta", "alpha"]).fit(verbose=0, seed=0, mstep=mstep, mstep_maxit=15, njobs=1)
        assert np.all(np.isfinite(res.m)), mstep


def test_uniform_prior_convergent_fit():
    from pyem.core.priors import UniformPrior
    params = _simulate_rw_params(10)
    sim = rw1a1b_sim(params, nblocks=2, ntrials=8)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
    prior = UniformPrior(lo=[-10, -10], hi=[10, 10])
    res = EMModel(all_data, rw1a1b_fit, ["beta", "alpha"]).fit(verbose=0, seed=0, prior=prior, mstep_maxit=15, njobs=1)
    assert np.all(np.isfinite(res.m))


def test_recover_fit_kwargs_passthrough_mstep():
    from pyem.models.rl import rw1a1b_model
    true_params = _simulate_rw_params(12)
    model = EMModel(all_data=None, fit_func=rw1a1b_model.fit, param_names=["beta", "alpha"],
                    simulate_func=rw1a1b_model.sim)
    r = model.recover(true_params, pr_inputs=["choices", "rewards"], nblocks=2, ntrials=10,
                      fit_kwargs={"seed": 0, "mstep": "laplace", "mstep_maxit": 15, "njobs": 1})
    assert np.all(np.isfinite(r["estimated_params"]))


def test_recover_fit_kwargs_forwarded_to_fit():
    # An invalid fit kwarg must reach EMModel.fit and raise -> proves fit_kwargs is forwarded.
    import pytest
    from pyem.models.rl import rw1a1b_model
    true_params = _simulate_rw_params(8)
    model = EMModel(all_data=None, fit_func=rw1a1b_model.fit, param_names=["beta", "alpha"],
                    simulate_func=rw1a1b_model.sim)
    with pytest.raises(TypeError):
        model.recover(true_params, pr_inputs=["choices", "rewards"], nblocks=2, ntrials=10,
                      fit_kwargs={"not_a_real_kwarg": 123, "mstep_maxit": 15, "njobs": 1})


def test_invalid_mstep_falls_back_to_user_prior(monkeypatch):
    from pyem.core import groupdist
    orig = groupdist.GaussianGroup.moments
    # force every M-step to report invalid (ok=0)
    monkeypatch.setattr(groupdist.GaussianGroup, "moments",
                        lambda self, h: (orig(self, h)[0], orig(self, h)[1], 0))
    params = _simulate_rw_params(6)
    sim = rw1a1b_sim(params, nblocks=1, ntrials=6)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
    res = EMModel(all_data, rw1a1b_fit, ["beta", "alpha"]).fit(verbose=0, seed=0, mstep_maxit=15, njobs=1)
    assert np.all(np.isfinite(res.m))   # invalid M-step every iter: fit runs without crashing and m stays finite


def test_fit_reproducible_on_fixed_data():
    # fit_kwargs seed makes the FIT deterministic given fixed data (what fit_kwargs enables).
    params = _simulate_rw_params(10)
    sim = rw1a1b_sim(params, nblocks=2, ntrials=10)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
    m1 = EMModel(all_data, rw1a1b_fit, ["beta", "alpha"]).fit(verbose=0, seed=0, mstep_maxit=15, njobs=1).m
    m2 = EMModel(all_data, rw1a1b_fit, ["beta", "alpha"]).fit(verbose=0, seed=0, mstep_maxit=15, njobs=1).m
    assert np.array_equal(m1, m2)


def test_glm_ar_fit():
    nsubjects, nparams, ntrials = 10, 3, 50
    rng = np.random.default_rng(0)
    true_params = rng.normal(size=(nsubjects, nparams - 1))
    true_params = np.hstack([true_params, rng.uniform(-0.5, 0.5, size=(nsubjects, 1))])  # add phi
    X, Y = glm_ar_sim(true_params, ntrials=ntrials)
    all_data = [[X[i], Y[i]] for i in range(nsubjects)]
    param_names = [f"b{i}" for i in range(nparams - 1)] + ["phi"]
    model = EMModel(all_data=all_data, fit_func=glm_ar_fit, param_names=param_names)
    res = model.fit(mstep_maxit=5, verbose=0, njobs=1)
    assert res.m.shape == (nparams, nsubjects)
    assert res.NPL.shape == (nsubjects,)


def test_recover_populates_outfit_and_all_data():
    import warnings
    import numpy as np
    from pyem.api import EMModel
    from pyem.models.rl import rw1a1b_model
    from test_helpers import _simulate_rw_params
    true_params = _simulate_rw_params(10)
    model = EMModel(all_data=None, fit_func=rw1a1b_model.fit, param_names=["beta", "alpha"],
                    simulate_func=rw1a1b_model.sim)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.recover(true_params, pr_inputs=["choices", "rewards"], nblocks=2, ntrials=10,
                      fit_kwargs={"seed": 0, "mstep_maxit": 5, "njobs": 1})
    # recover() warns that it overwrites all_data
    assert any("all_data" in str(wi.message) for wi in w)
    # all_data is now populated (simulated dataset), so get_outfit()/.outfit work
    assert model.all_data is not None and len(model.all_data) == 10
    of = model.get_outfit()
    assert isinstance(of, dict) and "nll" in of
    assert isinstance(model.outfit, dict) and "nll" in model.outfit
