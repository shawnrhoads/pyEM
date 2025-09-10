
import numpy as np
from pyem.api import EMModel
from pyem.models.rl import rw1a1b_simulate as rw_simulate, rw1a1b_fit as rw_fit
from pyem.core.compare import compare_models
from pyem.utils.stats import calc_BICint

def test_model_compare_basic():
    nsubjects, nblocks, ntrials = 4, 2, 8
    params = np.column_stack([np.random.randn(nsubjects), np.random.randn(nsubjects)])
    sim = rw_simulate(params, nblocks=nblocks, ntrials=ntrials)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
    m1 = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta","lr"])
    m2 = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta","lr"])  # dummy second
    r1 = m1.fit(mstep_maxit=5, verbose=0, njobs=1)
    r2 = m2.fit(mstep_maxit=5, verbose=0, njobs=1)
    rows = compare_models(
        [( "rw1", r1.__dict__, all_data, rw_fit ),
         ( "rw2", r2.__dict__, all_data, rw_fit )],
        bicint_kwargs={"nsamples": 10, "func_output":"all", "nll_key":"nll"},
        r2_kwargs={"ntrials": nblocks*ntrials, "nopts": 2}
    )
    assert len(rows) == 2
    assert any(r.LME is not None for r in rows)
