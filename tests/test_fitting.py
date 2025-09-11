import numpy as np
from pyem.api import EMModel
from pyem.models.rl import (
    rw1a1b_simulate, rw1a1b_fit,
    rw2a1b_simulate, rw2a1b_fit,
)
from pyem.models.bayes import simulate as bayes_simulate, fit as bayes_fit
from pyem.models.glm import simulate as glm_simulate, fit as glm_fit, simulate_decay, fit_decay


def test_rw1a1b_fit():
    nsubjects, nblocks, ntrials = 4, 2, 12
    params = np.random.randn(nsubjects, 2)
    sim = rw1a1b_simulate(params, nblocks=nblocks, ntrials=ntrials)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
    model = EMModel(all_data=all_data, fit_func=rw1a1b_fit, param_names=["beta", "alpha"])
    res = model.fit(mstep_maxit=5, verbose=0, njobs=1)
    assert res.m.shape == (2, nsubjects)
    assert res.NPL.shape == (nsubjects,)


def test_rw2a1b_fit():
    nsubjects, nblocks, ntrials = 4, 2, 12
    params = np.random.randn(nsubjects, 3)
    sim = rw2a1b_simulate(params, nblocks=nblocks, ntrials=ntrials)
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
    nsubjects, nblocks, ntrials = 3, 2, 10
    true_lambda = np.random.uniform(0.2, 0.8, size=(nsubjects, 1))
    sim = bayes_simulate(true_lambda, n_blocks=nblocks, n_trials=ntrials)
    all_data = [[sim["choices"][i], sim["observations"][i]] for i in range(nsubjects)]
    model = EMModel(all_data=all_data, fit_func=bayes_fit, param_names=["lambda"])
    res = model.fit(mstep_maxit=5, verbose=0, njobs=1)
    assert res.m.shape == (1, nsubjects)
    assert res.NPL.shape == (nsubjects,)


def test_glm_fit():
    nsubjects, nparams, ntrials = 3, 3, 50
    true_params = np.random.randn(nsubjects, nparams)
    X, Y = glm_simulate(true_params, ntrials=ntrials)
    all_data = [[X[i], Y[i]] for i in range(nsubjects)]
    model = EMModel(all_data=all_data, fit_func=glm_fit, param_names=[f"b{i}" for i in range(nparams)])
    res = model.fit(mstep_maxit=5, verbose=0, njobs=1)
    assert res.m.shape == (nparams, nsubjects)
    assert res.NPL.shape == (nsubjects,)


def test_glm_decay_fit():
    nsubjects, nparams, ntrials = 3, 2, 50
    true_params = np.random.randn(nsubjects, nparams + 1)  # last param is gamma
    X, Y = simulate_decay(true_params, ntrials=ntrials)
    all_data = [[X[i], Y[i]] for i in range(nsubjects)]
    param_names = [f"b{i}" for i in range(nparams)] + ["gamma"]
    model = EMModel(all_data=all_data, fit_func=fit_decay, param_names=param_names)
    res = model.fit(mstep_maxit=5, verbose=0, njobs=1)
    assert res.m.shape == (nparams + 1, nsubjects)
    assert res.NPL.shape == (nsubjects,)
