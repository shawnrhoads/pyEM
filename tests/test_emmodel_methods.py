"""Tests for enhanced EMModel methods and ModelComparison class."""

import numpy as np
import pytest

from pyem.api import EMModel
from pyem.core.compare import ModelComparison
from pyem.core.posterior import PCCResult
from pyem.models.rl import rw1a1b_fit as rw_fit, rw1a1b_simulate as rw_simulate
from pyem.utils.math import alpha2norm, beta2norm, norm2alpha, norm2beta


def _simulate_dataset(nsubjects: int, nblocks: int, ntrials: int) -> list[list]:
    params = np.column_stack([np.random.randn(nsubjects), np.random.randn(nsubjects)])
    sim = rw_simulate(params, nblocks=nblocks, ntrials=ntrials)
    return [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]


def test_compute_integrated_bic():
    nsubjects, nblocks, ntrials = 3, 2, 8
    all_data = _simulate_dataset(nsubjects, nblocks, ntrials)
    model = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta", "alpha"])
    model.fit(mstep_maxit=3, verbose=0, njobs=1)

    bicint = model.compute_integrated_bic(nsamples=5)
    assert np.isfinite(bicint)
    assert isinstance(bicint, float)


def test_compute_lme():
    nsubjects, nblocks, ntrials = 3, 2, 8
    all_data = _simulate_dataset(nsubjects, nblocks, ntrials)
    model = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta", "alpha"])
    model.fit(mstep_maxit=3, verbose=0, njobs=1)

    lap, lme, good = model.compute_lme()
    assert isinstance(lme, float)
    assert lap.shape == (nsubjects,)
    assert good.shape == (nsubjects,)


def test_get_outfit():
    nsubjects, nblocks, ntrials = 3, 2, 8
    all_data = _simulate_dataset(nsubjects, nblocks, ntrials)
    model = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta", "alpha"])
    model.fit(mstep_maxit=3, verbose=0, njobs=1)

    arrays = model.get_outfit()
    assert isinstance(arrays, dict)
    assert 'choices' in arrays
    assert 'rewards' in arrays
    assert 'nll' in arrays
    assert arrays['nll'].shape == (nsubjects,)
    assert 'choices_A' in arrays


def test_scipy_minimize():
    nsubjects, nblocks, ntrials = 3, 2, 8
    all_data = _simulate_dataset(nsubjects, nblocks, ntrials)
    model = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta", "alpha"])

    result = model.scipy_minimize()
    assert 'm' in result
    assert result['m'].shape == (2, nsubjects)
    assert result['individual_fit'] is True


def test_parameter_recovery():
    nsubjects, nblocks, ntrials = 3, 2, 8
    true_params = np.column_stack([np.random.randn(nsubjects), np.random.randn(nsubjects)])

    model = EMModel(all_data=None, fit_func=rw_fit, param_names=["beta", "alpha"], simulate_func=rw_simulate)
    recovery_dict = model.recover(true_params, nblocks=nblocks, ntrials=ntrials)

    assert 'true_params' in recovery_dict
    assert 'estimated_params' in recovery_dict
    assert 'correlation' in recovery_dict
    assert recovery_dict['true_params'].shape == recovery_dict['estimated_params'].shape
    assert recovery_dict['correlation'].shape == (true_params.shape[1],)

    fig = model.plot_recovery(recovery_dict)
    assert fig is not None


def test_parameter_transformations():
    model = EMModel(None, rw_fit, ["beta", "alpha"], param_xform=[norm2beta, norm2alpha])

    beta = model.param_xform[0](0.5)
    assert beta > 0

    alpha = model.param_xform[1](0.5)
    assert 0 < alpha < 1

    beta_func = model.get_param_transform("beta")
    assert beta_func is norm2beta

    alpha_func = model.get_param_transform("alpha")
    assert alpha_func is norm2alpha

    assert model.get_param_transform(0) is norm2beta
    assert model.get_param_transform(1) is norm2alpha

    original_alpha = 0.3
    norm_alpha = alpha2norm(original_alpha)
    recovered_alpha = norm2alpha(norm_alpha)
    assert np.isclose(original_alpha, recovered_alpha)

    original_beta = 5.0
    norm_beta = beta2norm(original_beta)
    recovered_beta = norm2beta(norm_beta)
    assert np.isclose(original_beta, recovered_beta)

    with pytest.raises(ValueError, match="param_xform length.*must match param_names length"):
        EMModel(None, rw_fit, ["beta", "alpha"], param_xform=[norm2beta])

    model_no_xform = EMModel(None, rw_fit, ["beta", "alpha"])
    with pytest.raises(ValueError, match="param_xform was not provided"):
        model_no_xform.get_param_transform("beta")

    with pytest.raises(ValueError, match="Parameter 'unknown' not found"):
        model.get_param_transform("unknown")

    with pytest.raises(ValueError, match="Index 5 out of range"):
        model.get_param_transform(5)


def test_subject_params_applies_param_xform():
    model = EMModel(None, rw_fit, ["beta", "alpha"], param_xform=[norm2beta, norm2alpha])
    model._out = {"m": np.array([[0.0, 1.0], [0.0, 1.0]])}

    params = model.subject_params()
    expected = np.array(
        [
            [norm2beta(0.0), norm2alpha(0.0)],
            [norm2beta(1.0), norm2alpha(1.0)],
        ]
    )
    assert np.allclose(params, expected)


def test_model_comparison_class():
    nsubjects, nblocks, ntrials = 3, 2, 8
    all_data = _simulate_dataset(nsubjects, nblocks, ntrials)

    model1 = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta", "alpha"], simulate_func=rw_simulate)
    model2 = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta", "alpha"], simulate_func=rw_simulate)

    model1.fit(mstep_maxit=3, verbose=0, njobs=1)
    model2.fit(mstep_maxit=3, verbose=0, njobs=1)

    comparison = ModelComparison([model1, model2], ['Model1', 'Model2'])
    results = comparison.compare()

    assert len(results) == 2
    assert all(hasattr(r, 'name') for r in results)
    assert all(hasattr(r, 'LME') for r in results)


def test_model_comparison_methods_exist():
    comparison = ModelComparison([], [])

    assert hasattr(comparison, 'compare')
    assert hasattr(comparison, 'identifiability_analysis')
    assert hasattr(comparison, 'plot_identifiability')
    assert callable(comparison.compare)
    assert callable(comparison.identifiability_analysis)
    assert callable(comparison.plot_identifiability)


def test_emmodel_basic_functionality():
    nsubjects, nblocks, ntrials = 4, 2, 8
    all_data = _simulate_dataset(nsubjects, nblocks, ntrials)

    model = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta", "alpha"])
    res = model.fit(mstep_maxit=5, verbose=0, njobs=1)

    assert res.m.shape == (2, nsubjects)
    assert res.NPL.shape == (nsubjects,)
    params_vary = np.var(res.m, axis=1) > 1e-6
    assert np.any(params_vary)


def test_pcc_pipeline():
    nsubjects, nblocks, ntrials = 3, 2, 6
    all_data = _simulate_dataset(nsubjects, nblocks, ntrials)

    model = EMModel(
        all_data=all_data,
        fit_func=rw_fit,
        param_names=["beta", "alpha"],
        simulate_func=rw_simulate,
    )

    with pytest.raises(RuntimeError):
        model.pcc("choices_A")

    fit_res = model.fit(mstep_maxit=4, verbose=0, njobs=1)
    assert fit_res.convergence in {True, False}

    pcc_result = model.pcc(
        "choices_A",
        n_sims=5,
        sim_kwargs={"nblocks": nblocks, "ntrials": ntrials},
    )
    assert isinstance(pcc_result, PCCResult)
    assert pcc_result.simulated.shape[0] == 5
    assert pcc_result.observed.shape[0] == nsubjects
    assert 0.0 <= pcc_result.p_value <= 1.0

    rewards_result = model.pcc(
        "rewards",
        n_sims=3,
        sim_kwargs={"nblocks": nblocks, "ntrials": ntrials},
    )
    assert rewards_result.observed.shape[1:] == (nblocks, ntrials)

    fig_single = model.plot_pcc(pcc_result, agent_index=0, show=False)
    assert fig_single is not None

    fig_all = model.plot_pcc(pcc_result, plot_all_agents=True, show=False)
    assert fig_all is not None

    fig_default = model.plot_pcc(show=False)
    assert fig_default is not None

