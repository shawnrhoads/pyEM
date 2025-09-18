import numpy as np
from scipy.stats import truncnorm, beta as beta_dist

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