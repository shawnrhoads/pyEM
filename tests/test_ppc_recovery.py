
import numpy as np
from pyem.api import EMModel
from pyem.models.rw import rw_simulate, rw_fit
from pyem.core.posterior import posterior_predictive_check, parameter_recovery

def stat_mean_reward(all_data):
    return float(np.mean([np.mean(b[1]) for b in all_data]))

def assemble(sim):
    return [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]

def test_ppc_and_recovery_smoke():
    nsubjects, nblocks, ntrials = 5, 2, 10
    true_params = np.column_stack([np.random.randn(nsubjects), np.random.randn(nsubjects)])
    sim = rw_simulate(true_params, nblocks=nblocks, ntrials=ntrials)
    all_data = assemble(sim)
    model = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta","lr"], simulate_func=rw_simulate)
    res = model.fit(mstep_maxit=5, verbose=0, njobs=1)
    ppc = posterior_predictive_check(model.simulate, res.__dict__, all_data, stat_fn=stat_mean_reward, n_sims=10, assemble_data=assemble)
    assert 0.0 <= ppc.p_value <= 1.0
    rec = parameter_recovery(true_params, res.m.T)
    assert rec.corr.shape[0] == 2
