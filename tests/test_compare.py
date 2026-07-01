
import numpy as np
import pytest
from pyem.api import EMModel
from pyem.models.rl import (
    rw1a1b_sim, rw1a1b_fit,
    rw1a1b_sim as rw_simulate, rw1a1b_fit as rw_fit,
    rw2a1b_sim, rw2a1b_fit,
)
from pyem.models.glm import glm_sim, glm_fit
from pyem.core.compare import compare_models, ModelComparison
from pyem.utils.stats import calc_BICint
from pyem.utils.math import norm2beta, norm2alpha
from test_helpers import _simulate_rw_params

def test_model_compare_basic():
    nsubjects, nblocks, ntrials = 4, 2, 8
    params = _simulate_rw_params(nsubjects)
    sim = rw_simulate(params, nblocks=nblocks, ntrials=ntrials)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
    m1 = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta", "alpha"])
    m2 = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta", "alpha"])  # dummy second
    r1 = m1.fit(mstep_maxit=5, verbose=0, njobs=1)
    r2 = m2.fit(mstep_maxit=5, verbose=0, njobs=1)
    rows = compare_models(
        [( "rw1", r1.__dict__, all_data, rw_fit ),
         ( "rw2", r2.__dict__, all_data, rw_fit )],
         ['rw1', 'rw2'],
        bicint_kwargs={"nsamples": 10, "func_output":"all", "nll_key":"nll"},
        r2_kwargs={"ntrials_total": nblocks*ntrials, "noptions": 2}
    )
    assert len(rows) == 2
    assert any(r.LME is not None for r in rows)


def test_identify_rw_models():
    """First real functional test of ModelComparison.identify() (previously
    only existence-checked). Confirms the round/simulate/fit/tally pipeline
    runs end-to-end and produces a structurally correct confusion table."""
    nblocks, ntrials, nrounds = 2, 8, 3
    # param_xform is required here: identify() draws raw Gaussian "true" params
    # and applies param_xform to map them into each model's natural-space bounds
    # before simulating — without it, rw*_sim's own bounds check rejects them.
    model1 = EMModel(all_data=None, fit_func=rw1a1b_fit, param_names=["beta", "alpha"],
                      param_xform=[norm2beta, norm2alpha],
                      simulate_func=rw1a1b_sim)
    model2 = EMModel(all_data=None, fit_func=rw2a1b_fit,
                      param_names=["beta", "alpha_pos", "alpha_neg"],
                      param_xform=[norm2beta, norm2alpha, norm2alpha],
                      simulate_func=rw2a1b_sim)
    mc = ModelComparison([model1, model2], ["RW1", "RW2"])
    df = mc.identify(
        mi_inputs=["choices", "rewards"],
        nrounds=nrounds,
        nsubjects=8,
        sim_kwargs={"nblocks": nblocks, "ntrials": ntrials},
        fit_kwargs={"mstep_maxit": 5, "njobs": 1, "seed": 0},
        seed=0,
    )
    expected_cols = {"Simulated", "Estimated", "LME", "BICint", "pseudoR2",
                      "bestlme", "bestbic", "bestR2"}
    assert expected_cols.issubset(df.columns)
    assert len(df) == 4  # 2 simulated x 2 estimated

    # every round produces a well-defined BICint winner for both models here,
    # so each Simulated model's winner tally must sum to nrounds
    for sim_name in ["RW1", "RW2"]:
        assert df.loc[df["Simulated"] == sim_name, "bestbic"].sum() == nrounds


def test_identify_requires_dict_output():
    """identify() is documented to require simulate_func to return a dict;
    GLM-family models return (X, Y) tuples and should fail loudly, not silently."""
    nsubjects, nparams, ntrials = 5, 2, 20
    true_params = np.random.default_rng(0).normal(size=(nsubjects, nparams))
    X, Y = glm_sim(true_params, ntrials=ntrials)
    all_data = [[X[i], Y[i]] for i in range(nsubjects)]
    glm_model_instance = EMModel(
        all_data=all_data, fit_func=glm_fit,
        param_names=[f"b{i}" for i in range(nparams)],
        simulate_func=glm_sim,
    )
    mc = ModelComparison([glm_model_instance], ["GLM"])
    with pytest.raises(ValueError, match="did not return a dict"):
        mc.identify(mi_inputs=["choices"], nrounds=1, nsubjects=5, sim_kwargs={"ntrials": ntrials})
