
import numpy as np
from scipy.stats import truncnorm, beta as beta_dist
from pyem.api import EMModel
from pyem.models.rl import rw1a1b_simulate as rw_simulate, rw1a1b_fit as rw_fit
from pyem.core.compare import compare_models
from pyem.utils.stats import calc_BICint

def test_model_compare_basic():
    nsubjects, nblocks, ntrials = 4, 2, 8
    betamin, betamax = .75, 10 # inverse temperature
    alphamin, alphamax = .05, .95 # learning rate
    beta_rv  = truncnorm((betamin-0)/1, (betamax-0)/1, loc=0, scale=2).rvs(nsubjects)
    a_lo, a_hi = beta_dist.cdf([alphamin, alphamax], 1.1, 1.1)
    alpha_rv = beta_dist.ppf(a_lo + np.random.rand(nsubjects)*(a_hi - a_lo), 1.1, 1.1)
    params = np.column_stack((beta_rv, alpha_rv))

    sim = rw_simulate(params, nblocks=nblocks, ntrials=ntrials)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
    m1 = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta", "alpha"])
    m2 = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta", "alpha"])  # dummy second
    r1 = m1.fit(mstep_maxit=5, verbose=0, njobs=1)
    r2 = m2.fit(mstep_maxit=5, verbose=0, njobs=1)
    rows = compare_models(
        [( "rw1", r1.__dict__, all_data, rw_fit ),
         ( "rw2", r2.__dict__, all_data, rw_fit )],
         ['rw1', 'rw2'],
        bicint_kwargs={"nsamples": 10, "func_output":"all", "nll_key":"nll"},
        r2_kwargs={"ntrials_total": nblocks*ntrials, "noptions": 2}
    )
    assert len(rows) == 2
    assert any(r.LME is not None for r in rows)
