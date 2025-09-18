import numpy as np
from pyem.api import EMModel
from pyem.core.priors import GaussianPrior
from pyem.models.rl import rw1a1b_fit as rw_fit, rw1a1b_simulate as rw_simulate


def test_uniform_prior_and_fit():
    nsubjects, nblocks, ntrials = 2, 1, 4
    params = np.column_stack([np.zeros(nsubjects), np.zeros(nsubjects)])
    sim = rw_simulate(params, nblocks=nblocks, ntrials=ntrials)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]

    prior = GaussianPrior(mu=[0.0, 0.0], sigma=[100.0, 100.0])
    model = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta", "alpha"])
    result = model.fit(prior=prior, verbose=0)

    assert result.m.shape == (2, nsubjects)
    assert np.isfinite(prior.logpdf([0.0, 0.0]))

