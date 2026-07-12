"""
Direct unit tests for pyem.utils.stats — previously only exercised indirectly
through a full model fit. These pin down the Laplace-approximation (LME) and
integrated-BIC formulas against independently hand-derived expected values.
"""
import numpy as np
from pyem.utils.stats import calc_LME, calc_BICint


def test_calc_lme_identity_hessian():
    """With an identity Hessian, log|H| = 0, isolating the rest of the formula:

        Lap_i = -NPL_i - 0.5*log|H| + (d/2)*log(2*pi)
              = -NPL_i + (d/2)*log(2*pi)                  (since log|H| = 0)
        lme   = sum(Lap) - d*log(nsubjects)

    Derived independently from the standard Laplace approximation to log
    model evidence (see e.g. Daw, 2011), not by re-reading calc_LME's source.
    """
    nparams, nsubjects = 2, 3
    inv_h = np.stack([np.eye(nparams)] * nsubjects, axis=-1)  # H = inv_h = identity
    NPL = np.array([1.0, 2.0, 3.0])

    Lap, lme, good = calc_LME(inv_h, NPL)

    expected_Lap = -NPL + (nparams / 2) * np.log(2 * np.pi)
    expected_lme = np.sum(expected_Lap) - nparams * np.log(nsubjects)

    assert np.allclose(Lap, expected_Lap)
    assert np.isclose(lme, expected_lme)
    assert np.all(good == 1)


def test_calc_lme_nonidentity_hessian_sign():
    """A non-identity Hessian exercises the logdet term's sign.

    With inv_h = diag([4, 1]) for a single subject, log|inv_h| = log(4).
    Lap = -NPL - 0.5*(-log|inv_h|) + (d/2)*log(2*pi) = -NPL + 0.5*log(4) + log(2*pi).
    """
    inv_h = np.diag([4.0, 1.0])[:, :, None]  # shape (2, 2, 1)
    NPL = np.array([0.0])

    Lap, lme, good = calc_LME(inv_h, NPL)

    expected = -0.0 + 0.5 * np.log(4.0) + (2 / 2) * np.log(2 * np.pi)
    assert np.isclose(Lap[0], expected)
    assert good[0] == 1


def test_calc_bicint_trial_counting_not_doubled():
    """Regression test: calc_BICint used to sum .size across every array-like
    field of the first subject's data (e.g. choices.size + rewards.size),
    double-counting trials for the common [choices, rewards]-shaped data used
    by the RW/Bayes model families. It should now use only the first field.
    """
    nblocks, ntrials, npar = 2, 6, 2
    fixed_nll = 5.0

    def stub_fit_func(params, choices, rewards, output="all"):
        return {"nll": fixed_nll}

    all_data = [[np.zeros((nblocks, ntrials)), np.zeros((nblocks, ntrials))]]
    mu = np.zeros(npar)
    sigma = np.ones(npar)

    bicint = calc_BICint(all_data, ["p1", "p2"], mu, sigma, stub_fit_func, nsamples=5)

    # stub always returns the same nll regardless of sampled params, so the
    # Monte Carlo integration collapses to an exact value: iLog = -fixed_nll.
    expected_correct = -2 * (-fixed_nll) + npar * np.log(nblocks * ntrials)
    expected_old_buggy = -2 * (-fixed_nll) + npar * np.log(2 * nblocks * ntrials)

    assert np.isclose(bicint, expected_correct)
    assert not np.isclose(bicint, expected_old_buggy)


def test_calc_bicint_ntrials_total_override():
    """Explicit ntrials_total should override auto-detection entirely."""
    nblocks, ntrials, npar = 2, 6, 2
    fixed_nll = 5.0

    def stub_fit_func(params, choices, rewards, output="all"):
        return {"nll": fixed_nll}

    all_data = [[np.zeros((nblocks, ntrials)), np.zeros((nblocks, ntrials))]]
    mu = np.zeros(npar)
    sigma = np.ones(npar)

    bicint = calc_BICint(
        all_data, ["p1", "p2"], mu, sigma, stub_fit_func, nsamples=5, ntrials_total=100
    )
    expected = -2 * (-fixed_nll) + npar * np.log(100)
    assert np.isclose(bicint, expected)


def test_calc_bicint_njobs_serial():
    from pyem.models.rl_mf import rw1a1b_sim, rw1a1b_fit
    sim = rw1a1b_sim(np.array([[3.0, 0.5], [2.0, 0.4]]), nblocks=2, ntrials=10, seed=0)
    all_data = [[sim["choices"][s], sim["rewards"][s]] for s in range(2)]
    mu = np.array([0.0, 0.0])
    sigma = np.array([1.0, 1.0])
    v = calc_BICint(all_data, ["beta", "alpha"], mu, sigma, rw1a1b_fit,
                    nsamples=40, njobs=1)
    assert np.isfinite(v)
