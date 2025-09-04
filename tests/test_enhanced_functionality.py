"""
Tests for enhanced EMModel methods and ModelComparison class.
"""
import numpy as np
import pytest
from pyem.api import EMModel
from pyem.models.rw import rw_simulate, rw_fit
from pyem.core.compare import ModelComparison


def test_compute_integrated_bic():
    """Test compute_integrated_bic method."""
    nsubjects, nblocks, ntrials = 3, 2, 8
    params = np.column_stack([np.random.randn(nsubjects), np.random.randn(nsubjects)])
    sim = rw_simulate(params, nblocks=nblocks, ntrials=ntrials)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
    model = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta","lr"])
    model.fit(mstep_maxit=3, verbose=0, njobs=1)
    
    bicint = model.compute_integrated_bic(nsamples=5)
    assert np.isfinite(bicint)
    assert isinstance(bicint, float)


def test_compute_lme():
    """Test compute_lme method."""
    nsubjects, nblocks, ntrials = 3, 2, 8
    params = np.column_stack([np.random.randn(nsubjects), np.random.randn(nsubjects)])
    sim = rw_simulate(params, nblocks=nblocks, ntrials=ntrials)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
    model = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta","lr"])
    model.fit(mstep_maxit=3, verbose=0, njobs=1)
    
    lap, lme, good = model.compute_lme()
    assert isinstance(lme, float)
    assert lap.shape == (nsubjects,)
    assert good.shape == (nsubjects,)


def test_calculate_final_arrays():
    """Test calculate_final_arrays method."""
    nsubjects, nblocks, ntrials = 3, 2, 8
    params = np.column_stack([np.random.randn(nsubjects), np.random.randn(nsubjects)])
    sim = rw_simulate(params, nblocks=nblocks, ntrials=ntrials)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
    model = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta","lr"])
    model.fit(mstep_maxit=3, verbose=0, njobs=1)
    
    arrays = model.calculate_final_arrays()
    assert 'choices' in arrays
    assert 'rewards' in arrays
    assert 'choice_nll' in arrays
    assert arrays['choice_nll'].shape == (nsubjects,)


def test_fit_individual_nll():
    """Test fit_individual_nll method."""
    nsubjects, nblocks, ntrials = 3, 2, 8
    params = np.column_stack([np.random.randn(nsubjects), np.random.randn(nsubjects)])
    sim = rw_simulate(params, nblocks=nblocks, ntrials=ntrials)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
    model = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta","lr"])
    
    # Test with EMfit
    result_em = model.fit_individual_nll(use_emfit=True)
    assert 'm' in result_em
    assert result_em['m'].shape == (2, nsubjects)
    
    # Test without EMfit (individual fitting)
    result_indiv = model.fit_individual_nll(use_emfit=False)
    assert 'm' in result_indiv
    assert result_indiv['m'].shape == (2, nsubjects)
    assert result_indiv['individual_fit'] == True


def test_parameter_recovery():
    """Test parameter recovery functionality."""
    nsubjects, nblocks, ntrials = 3, 2, 8
    true_params = np.column_stack([np.random.randn(nsubjects), np.random.randn(nsubjects)])
    
    model = EMModel(all_data=None, fit_func=rw_fit, param_names=["beta","lr"], simulate_func=rw_simulate)
    recovery_dict = model.recover(true_params, nblocks=nblocks, ntrials=ntrials)
    
    assert 'true_params' in recovery_dict
    assert 'estimated_params' in recovery_dict
    assert 'correlation' in recovery_dict
    assert 'rmse' in recovery_dict
    assert 'mae' in recovery_dict
    assert recovery_dict['true_params'].shape == recovery_dict['estimated_params'].shape
    
    # Test plot_recovery
    fig = model.plot_recovery(recovery_dict)
    assert fig is not None


def test_parameter_transformations():
    """Test parameter transformation functions as class attributes."""
    model = EMModel(None, rw_fit, ["beta","lr"])
    
    # Test norm2alpha
    alpha = model.norm2alpha(0.5)
    assert 0 < alpha < 1
    
    # Test norm2beta  
    beta = model.norm2beta(0.5)
    assert beta > 0
    
    # Test roundtrip transformations
    original_alpha = 0.3
    norm_alpha = model.alpha2norm(original_alpha)
    recovered_alpha = model.norm2alpha(norm_alpha)
    assert np.isclose(original_alpha, recovered_alpha)
    
    original_beta = 5.0
    norm_beta = model.beta2norm(original_beta)
    recovered_beta = model.norm2beta(norm_beta)
    assert np.isclose(original_beta, recovered_beta)


def test_model_comparison_class():
    """Test ModelComparison class."""
    nsubjects, nblocks, ntrials = 3, 2, 8
    params = np.column_stack([np.random.randn(nsubjects), np.random.randn(nsubjects)])
    sim = rw_simulate(params, nblocks=nblocks, ntrials=ntrials)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
    
    # Create two models
    model1 = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta","lr"], simulate_func=rw_simulate)
    model2 = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta","lr"], simulate_func=rw_simulate)
    
    # Fit models
    model1.fit(mstep_maxit=3, verbose=0, njobs=1)
    model2.fit(mstep_maxit=3, verbose=0, njobs=1)
    
    # Test comparison
    comparison = ModelComparison([model1, model2], ['Model1', 'Model2'])
    results = comparison.compare()
    
    assert len(results) == 2
    assert all(hasattr(r, 'name') for r in results)
    assert all(hasattr(r, 'LME') for r in results)


def test_model_comparison_methods_exist():
    """Test that ModelComparison has the expected methods."""
    comparison = ModelComparison([], [])
    
    # Test methods exist
    assert hasattr(comparison, 'compare')
    assert hasattr(comparison, 'identifiability_analysis')
    assert hasattr(comparison, 'plot_identifiability')
    assert callable(comparison.compare)
    assert callable(comparison.identifiability_analysis)
    assert callable(comparison.plot_identifiability)


def test_emmodel_without_shared_mask():
    """Test that EMModel works without shared_mask functionality."""
    nsubjects, nblocks, ntrials = 4, 2, 8
    params = np.column_stack([np.random.randn(nsubjects), np.random.randn(nsubjects)])
    sim = rw_simulate(params, nblocks=nblocks, ntrials=ntrials)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
    
    model = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta","lr"])
    
    # Should work without shared_mask parameter
    res = model.fit(mstep_maxit=5, verbose=0, njobs=1)
    assert res.m.shape == (2, nsubjects)
    assert res.NPL.shape == (nsubjects,)
    
    # Parameters should be different across subjects (no shared constraints)
    params_vary = np.var(res.m, axis=1) > 1e-6
    assert np.any(params_vary)  # At least some parameters should vary across subjects