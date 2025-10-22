import numpy as np
from pyem.api import EMModel
from pyem.models.rl import (
    rw1a1b_sim, rw1a1b_fit,
    rw2a1b_sim, rw2a1b_fit,
)
from pyem.models.bayes import bayes_sim, bayes_fit
from pyem.models.glm import glm_sim, glm_fit, glm_decay_sim, glm_decay_fit
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
