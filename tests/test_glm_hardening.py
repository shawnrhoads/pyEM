import numpy as np
import pytest
from pyem.models.glm import glm_sim, glm_fit


def test_glm_fit_unknown_output_raises():
    X, Y = glm_sim(np.zeros((1, 3)), ntrials=20)
    with pytest.raises(ValueError):
        glm_fit(np.zeros(3), X[0], Y[0], output="bogus")


def test_glm_fit_perfect_fit_finite():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(30), rng.normal(size=30)])
    beta = np.array([1.0, 2.0])
    Y = X.dot(beta)  # exact linear -> residual std 0
    out = glm_fit(beta, X, Y, output="all")
    assert np.isfinite(out["nll"])


def test_glm_sim_seed_reproducible():
    a = glm_sim(np.zeros((2, 3)), ntrials=10, seed=5)
    b = glm_sim(np.zeros((2, 3)), ntrials=10, seed=5)
    assert np.array_equal(a[1], b[1])            # same seed -> identical Y
    c = glm_sim(np.zeros((2, 3)), ntrials=10, seed=6)
    assert not np.array_equal(a[1], c[1])        # different seed -> different Y (seed truly used)


import pytest as _pytest
from pyem.models.glm import (glm_fit, glm_decay_fit, logit_fit, logit_decay_fit,
                             glm_ar_fit, glm_sim, glm_decay_sim)

@_pytest.mark.parametrize("fit", [glm_fit, glm_decay_fit, logit_fit, logit_decay_fit, glm_ar_fit])
def test_all_fits_raise_on_unknown_output(fit):
    X, Y = glm_sim(np.zeros((1, 3)), ntrials=20)
    # X has 3 predictor columns. glm_fit/logit_fit take a length-3 weight vector.
    # The decay fits append a gamma param, and glm_ar_fit appends a phi param, so
    # both need length 4 to line up with X's 3 columns and reach the output check.
    params = np.zeros(4) if fit in (glm_decay_fit, logit_decay_fit, glm_ar_fit) else np.zeros(3)
    with _pytest.raises(ValueError):
        fit(params, X[0], Y[0], output="bogus")

def test_glm_decay_sim_decay_window_differs():
    # 'twostep' (3 terms) vs 'onestep' (2 terms) must produce different Y for the same seed
    p = np.array([[0.5, 1.0, 0.5]])  # weights (incl intercept) + gamma
    a = glm_decay_sim(p, ntrials=30, decay='twostep', seed=1)
    b = glm_decay_sim(p, ntrials=30, decay='onestep', seed=1)
    assert not np.array_equal(a[1], b[1])
