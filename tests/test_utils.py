import numpy as np
from scipy.stats import truncnorm, beta as beta_dist
from pyem.core.posterior import parameter_recovery, model_identifiability
from pyem.api import EMModel
from pyem.models.rl import (
    rw1a1b_simulate, rw1a1b_fit,
    rw2a1b_simulate, rw2a1b_fit,
)


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

def test_parameter_recovery_function():
    true = np.array([[0.0, 1.0], [1.0, 0.0]])
    est = true + 0.1
    res = parameter_recovery(true, est)
    assert res.corr.shape == (2,)
    assert np.allclose(res.rmse, 0.1, atol=1e-6)


def test_model_identifiability():
    # two simple RL models
    cand1 = EMModel(all_data=None, fit_func=rw1a1b_fit, param_names=["beta", "alpha"],
                    simulate_func=rw1a1b_simulate)
    cand2 = EMModel(all_data=None, fit_func=rw2a1b_fit,
                    param_names=["beta", "alpha_pos", "alpha_neg"],
                    simulate_func=rw2a1b_simulate)
    models = [cand1, cand2]
    params1 = _simulate_rw_params(10, nparams=2)
    params2 = _simulate_rw_params(10, nparams=3)
    param_sets = [(cand1, params1), (cand2, params2)]
    chooser = lambda out: np.sum(out["NPL"])
    res = model_identifiability(models, param_sets, nblocks=1, ntrials=5, chooser=chooser)
    assert res.confusion.shape == (2, 2)
    assert np.all(res.confusion.sum(axis=1) == 1)
