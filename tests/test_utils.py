import numpy as np
from pyem.core.posterior import parameter_recovery, model_identifiability
from pyem.api import EMModel
from pyem.models.rl import (
    rw1a1b_simulate, rw1a1b_fit,
    rw2a1b_simulate, rw2a1b_fit,
)
from test_helpers import _simulate_rw_params

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
