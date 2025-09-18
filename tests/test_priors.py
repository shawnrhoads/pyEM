import numpy as np
from scipy.stats import truncnorm, beta as beta_dist
from pyem.api import EMModel
from pyem.core.priors import GaussianPrior
from pyem.models.rl import rw1a1b_fit as rw_fit, rw1a1b_simulate as rw_simulate

def _simulate_rw_params(
    nsubjects: int,
    betamin: float = 0.75,
    betamax: float = 10.0,
    alphamin: float = 0.05,
    alphamax: float = 0.95,
    a: float = 1.1,  # beta-dist shape
    b: float = 1.1,  # beta-dist shape
    seed: int | None = 0,  # keep tests reproducible; set None to randomize
) -> np.ndarray:
    """
    Return array of shape (nsubjects, 2) with columns [beta, alpha].

    beta ~ truncated normal (loc=0, scale=2) restricted to [betamin, betamax]
    alpha ~ beta(a, b) restricted to [alphamin, alphamax] via inverse-CDF sampling
    """
    rng = np.random.default_rng(seed)

    # beta (inverse temperature)
    tn = truncnorm((betamin - 0) / 1, (betamax - 0) / 1, loc=0, scale=2)
    beta_rv = tn.rvs(nsubjects, random_state=rng)

    # alpha (learning rate), truncated via CDF window
    a_lo, a_hi = beta_dist.cdf([alphamin, alphamax], a, b)
    u = a_lo + rng.random(nsubjects) * (a_hi - a_lo)
    alpha_rv = beta_dist.ppf(u, a, b)

    return np.column_stack((beta_rv, alpha_rv))


def test_uniform_prior_and_fit():
    nsubjects, nblocks, ntrials = 10, 1, 4
    params = _simulate_rw_params(nsubjects)
    sim = rw_simulate(params, nblocks=nblocks, ntrials=ntrials)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]

    prior = GaussianPrior(mu=[0.0, 0.0], sigma=[100.0, 100.0])
    model = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta", "alpha"])
    result = model.fit(prior=prior, verbose=0)

    assert result.m.shape == (2, nsubjects)
    assert np.isfinite(prior.logpdf([0.0, 0.0]))

