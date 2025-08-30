
import numpy as np
from pyem.api import EMModel
from pyem.models.rw import rw_simulate, rw_fit

def test_smoke_fit():
    nsubjects, nblocks, ntrials = 6, 2, 12
    params = np.column_stack([np.random.randn(nsubjects), np.random.randn(nsubjects)])
    sim = rw_simulate(params, nblocks=nblocks, ntrials=ntrials)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
    model = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta","lr"])
    res = model.fit(mstep_maxit=5, verbose=0, njobs=1)
    assert res.m.shape == (2, nsubjects)
    assert res.NPL.shape == (nsubjects,)
