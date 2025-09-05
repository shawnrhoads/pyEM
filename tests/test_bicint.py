
import numpy as np
from pyem.api import EMModel
from pyem.models.rl import rw1a1b_simulate as rw_simulate, rw1a1b_fit as rw_fit
from pyem.utils.stats import calc_BICint

def test_bicint_smoke():
    nsubjects, nblocks, ntrials = 3, 2, 8
    params = np.column_stack([np.random.randn(nsubjects), np.random.randn(nsubjects)])
    sim = rw_simulate(params, nblocks=nblocks, ntrials=ntrials)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
    model = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta","lr"])
    res = model.fit(mstep_maxit=5, verbose=0, njobs=1)
    bicint = calc_BICint(all_data, ["beta","lr"], res.posterior_mu, res.posterior_sigma, rw_fit, nsamples=5, func_output="all", nll_key="NLL")
    assert np.isfinite(bicint)
