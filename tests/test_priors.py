import numpy as np

from pyem.api import EMModel
from pyem.core.priors import GaussianPrior
from pyem.models.rl import rw1a1b_fit as rw_fit, rw1a1b_simulate as rw_simulate


def test_gaussian_prior_and_fit():
    nsubjects, nblocks, ntrials = 2, 1, 4
    params = np.column_stack([np.zeros(nsubjects), np.zeros(nsubjects)])
    sim = rw_simulate(params, nblocks=nblocks, ntrials=ntrials)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]

    prior = GaussianPrior(mu=np.zeros(2), sigma=np.ones(2))
    model = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta", "alpha"])
    result = model.fit(prior=prior, verbose=0, njobs=1)

    assert result.m.shape == (2, nsubjects)
    assert np.isfinite(prior.logpdf([0.0, 0.0]))
    assert prior.logpdf([10.0, 0.0]) < prior.logpdf([0.0, 0.0])


def test_gaussian_prior_default_factory():
    nparams = 4
    prior = GaussianPrior.default(nparams, seed=42)

    assert isinstance(prior, GaussianPrior)
    assert prior.mu.shape == (nparams,)
    assert prior.sigma.shape == (nparams,)
    assert np.all(prior.sigma > 0)

    prior_again = GaussianPrior.default(nparams, seed=42)
    np.testing.assert_allclose(prior.mu, prior_again.mu)
    np.testing.assert_allclose(prior.sigma, prior_again.sigma)

