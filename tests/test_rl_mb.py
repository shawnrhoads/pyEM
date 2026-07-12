import numpy as np
import pytest
from pyem.models.rl_mb import (
    sarsa_lambda_sim, sarsa_lambda_fit,
    model_based_sim, model_based_fit,
    hybrid_mbmf_sim, hybrid_mbmf_fit,
)


@pytest.mark.parametrize("sim,fit,nat", [
    (sarsa_lambda_sim, sarsa_lambda_fit, np.array([[3.0, 3.0, 0.5, 0.5, 0.5, 0.0]])),
    (model_based_sim, model_based_fit, np.array([[3.0, 3.0, 0.5, 0.0]])),
    (hybrid_mbmf_sim, hybrid_mbmf_fit, np.array([[3.0, 3.0, 0.5, 0.5, 0.5, 0.5, 0.0]])),
])
def test_twostep_roundtrip(sim, fit, nat):
    out = sim(nat, ntrials=60, seed=0)
    nparams = nat.shape[1]
    res = fit(np.zeros(nparams), out["choices1"][0], out["states2"][0],
              out["choices2"][0], out["rewards"][0], output="all")
    assert np.isfinite(res["nll"])
