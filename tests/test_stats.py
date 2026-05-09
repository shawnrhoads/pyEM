import numpy as np
import pytest

from pyem.utils.stats import (
    calc_BICint,
    overall_predictive_probability_from_nll,
    pseudo_r2_from_nll,
)


def _dummy_fit_func(pars, choices, rewards, output="all"):
    del pars, output
    return {"nll": float(choices.size), "rewards_sum": float(np.sum(rewards))}


def test_calc_bicint_uses_total_trials_across_subjects():
    all_data = [
        (np.array([0, 1, 0]), np.array([1, 0, 1])),
        (np.array([1, 1]), np.array([0, 1])),
    ]
    mu = np.array([0.1, -0.1])
    sigma = np.zeros(2)

    bicint = calc_BICint(
        all_data=all_data,
        param_names=["beta", "alpha"],
        mu=mu,
        sigma=sigma,
        fit_func=_dummy_fit_func,
        nsamples=5,
        func_output="all",
        nll_key="nll",
    )

    total_trials = 5
    expected = -2 * (-(3 + 2)) + 2 * np.log(total_trials)
    assert np.isclose(bicint, expected)


def test_calc_bicint_is_numerically_stable_with_large_nll():
    def huge_nll_fit(pars, choices, rewards, output="all"):
        del pars, choices, rewards, output
        return {"nll": 1_000.0}

    all_data = [
        (np.array([0, 1, 0]), np.array([1, 0, 1])),
    ]

    bicint = calc_BICint(
        all_data=all_data,
        param_names=["beta"],
        mu=np.array([0.0]),
        sigma=np.zeros(1),
        fit_func=huge_nll_fit,
        nsamples=10,
        func_output="all",
        nll_key="nll",
    )

    assert np.isfinite(bicint)


def test_pseudo_r2_rejects_invalid_metric():
    with pytest.raises(ValueError, match="metric must be 'median' or 'mean'"):
        pseudo_r2_from_nll(np.array([2.0, 3.0]), ntrials_total=10, noptions=2, metric="mode")


def test_overall_predictive_probability_from_nll_matches_formula():
    nll = np.array([2.0, 3.0])
    nchoices_total = 10
    expected = np.exp(-(2.0 + 3.0) / nchoices_total)
    got = overall_predictive_probability_from_nll(nll, nchoices_total)
    assert np.isclose(got, expected)


def test_overall_predictive_probability_from_nll_return_log_and_validation():
    nll = np.array([1.0, 2.0, np.nan])
    log_val = overall_predictive_probability_from_nll(nll, nchoices_total=6, return_log=True)
    assert np.isclose(log_val, -0.5)

    with pytest.raises(ValueError, match="nchoices_total must be a positive integer"):
        overall_predictive_probability_from_nll(np.array([1.0]), nchoices_total=0)
