import numpy as np
from pyem.core.posterior import parameter_recovery

def test_parameter_recovery_function():
    true = np.array([[0.0, 1.0], [1.0, 0.0]])
    est = true + 0.1
    res = parameter_recovery(true, est)
    assert res.corr.shape == (2,)
    assert np.allclose(res.rmse, 0.1, atol=1e-6)
