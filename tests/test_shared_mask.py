
import numpy as np
from pyem.api import EMModel
from pyem.models.rw import rw_simulate, rw_fit

def test_shared_mask_makes_param_equal():
    nsubjects, nblocks, ntrials = 4, 2, 8
    params = np.column_stack([np.zeros(nsubjects), np.random.randn(nsubjects)])  # zero beta norm
    sim = rw_simulate(params, nblocks=nblocks, ntrials=ntrials)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
    model = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta","lr"])
    res = model.fit(mstep_maxit=5, verbose=0, njobs=1, shared_mask=np.array([True, False]))
    # beta should be equal across subjects
    betas = res.m[0, :]
    assert np.allclose(betas, betas[0])
