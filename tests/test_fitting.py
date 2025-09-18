import numpy as np
from scipy.stats import truncnorm, beta as beta_dist
from pyem.api import EMModel
from pyem.models.rl import (
    rw1a1b_simulate, rw1a1b_fit,
    rw2a1b_simulate, rw2a1b_fit,
)
from pyem.models.bayes import simulate as bayes_simulate, fit as bayes_fit
from pyem.models.glm import simulate as glm_simulate, fit as glm_fit, simulate_decay, fit_decay


# ---------------------------------------------------------------------
# Helper: simulate subject-level RW parameters once and reuse everywhere
# ---------------------------------------------------------------------
def _simulate_rw_params(
    nsubjects: int,
    nparams: int = 2,
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
    if nparams == 2:
        a_lo, a_hi = beta_dist.cdf([alphamin, alphamax], a, b)
        u = a_lo + rng.random(nsubjects) * (a_hi - a_lo)
        alpha_rv = beta_dist.ppf(u, a, b)

        return np.column_stack((beta_rv, alpha_rv))
    elif nparams == 3:
        a_lo, a_hi = beta_dist.cdf([alphamin, alphamax], a, b)
        u1 = a_lo + rng.random(nsubjects) * (a_hi - a_lo)
        u2 = a_lo + rng.random(nsubjects) * (a_hi - a_lo)
        alpha_pos_rv = beta_dist.ppf(u1, a, b)
        alpha_neg_rv = beta_dist.ppf(u2, a, b)

        return np.column_stack((beta_rv, alpha_pos_rv, alpha_neg_rv))

def test_rw1a1b_fit():
    nsubjects, nblocks, ntrials = 10, 2, 12
    params = _simulate_rw_params(nsubjects)
    sim = rw1a1b_simulate(params, nblocks=nblocks, ntrials=ntrials)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
    model = EMModel(all_data=all_data, fit_func=rw1a1b_fit, param_names=["beta", "alpha"])
    res = model.fit(mstep_maxit=5, verbose=0, njobs=1)
    assert res.m.shape == (2, nsubjects)
    assert res.NPL.shape == (nsubjects,)


def test_rw2a1b_fit():
    nsubjects, nblocks, ntrials = 10, 2, 12
    params = _simulate_rw_params(nsubjects, 3)
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
    nsubjects, nblocks, ntrials = 10, 2, 10
    true_lambda = np.random.uniform(0.2, 0.8, size=(nsubjects, 1))
    sim = bayes_simulate(true_lambda, n_blocks=nblocks, n_trials=ntrials)
    all_data = [[sim["choices"][i], sim["observations"][i]] for i in range(nsubjects)]
    model = EMModel(all_data=all_data, fit_func=bayes_fit, param_names=["lambda"])
    res = model.fit(mstep_maxit=5, verbose=0, njobs=1)
    assert res.m.shape == (1, nsubjects)
    assert res.NPL.shape == (nsubjects,)


def test_glm_fit():
    nsubjects, nparams, ntrials = 10, 3, 50
    true_params = np.random.randn(nsubjects, nparams)
    X, Y = glm_simulate(true_params, ntrials=ntrials)
    all_data = [[X[i], Y[i]] for i in range(nsubjects)]
    model = EMModel(all_data=all_data, fit_func=glm_fit, param_names=[f"b{i}" for i in range(nparams)])
    res = model.fit(mstep_maxit=5, verbose=0, njobs=1)
    assert res.m.shape == (nparams, nsubjects)
    assert res.NPL.shape == (nsubjects,)


def test_glm_decay_fit():
    nsubjects, nparams, ntrials = 10, 2, 50
    true_params = np.random.randn(nsubjects, nparams + 1)  # all random normal
    true_params[:, -1] = np.random.uniform(0, 1, size=nsubjects)  # overwrite last param with uniform(0,1)
    X, Y = simulate_decay(true_params, ntrials=ntrials)
    all_data = [[X[i], Y[i]] for i in range(nsubjects)]
    param_names = [f"b{i}" for i in range(nparams)] + ["gamma"]
    model = EMModel(all_data=all_data, fit_func=fit_decay, param_names=param_names)
    res = model.fit(mstep_maxit=5, verbose=0, njobs=1)
    assert res.m.shape == (nparams + 1, nsubjects)
    assert res.NPL.shape == (nsubjects,)
