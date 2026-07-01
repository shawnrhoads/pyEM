"""
Tests for the ModelSpec wrapper objects (<name>_model) defined alongside each
model's <name>_sim/<name>_fit functions in pyem.models.rl/glm/bayes.
"""
import pytest
from pyem.core.modelspec import ModelSpec
from pyem.models.rl import (
    rw1a1b_sim, rw1a1b_fit, rw1a1b_model,
    rw2a1b_sim, rw2a1b_fit, rw2a1b_model,
    rw3a1b_sim, rw3a1b_fit, rw3a1b_model,
    rw4a1b_sim, rw4a1b_fit, rw4a1b_model,
)
from pyem.models.glm import (
    glm_sim, glm_fit, glm_model,
    glm_decay_sim, glm_decay_fit, glm_decay_model,
    logit_sim, logit_fit, logit_model,
    logit_decay_sim, logit_decay_fit, logit_decay_model,
    glm_ar_sim, glm_ar_fit, glm_ar_model,
)
from pyem.models.bayes import bayes_sim, bayes_fit, bayes_model

MODEL_CASES = [
    (rw1a1b_model, rw1a1b_sim, rw1a1b_fit, "rw1a1b"),
    (rw2a1b_model, rw2a1b_sim, rw2a1b_fit, "rw2a1b"),
    (rw3a1b_model, rw3a1b_sim, rw3a1b_fit, "rw3a1b"),
    (rw4a1b_model, rw4a1b_sim, rw4a1b_fit, "rw4a1b"),
    (glm_model, glm_sim, glm_fit, "glm"),
    (glm_decay_model, glm_decay_sim, glm_decay_fit, "glm_decay"),
    (logit_model, logit_sim, logit_fit, "logit"),
    (logit_decay_model, logit_decay_sim, logit_decay_fit, "logit_decay"),
    (glm_ar_model, glm_ar_sim, glm_ar_fit, "glm_ar"),
    (bayes_model, bayes_sim, bayes_fit, "bayes"),
]


@pytest.mark.parametrize("model, sim_fn, fit_fn, expected_id", MODEL_CASES)
def test_modelspec_attributes(model, sim_fn, fit_fn, expected_id):
    assert isinstance(model, ModelSpec)
    assert model.id == expected_id
    assert isinstance(model.desc, str) and len(model.desc) > 0
    assert isinstance(model.spec, dict) and len(model.spec) > 0
    assert callable(model.sim)
    assert callable(model.fit)


@pytest.mark.parametrize("model, sim_fn, fit_fn, expected_id", MODEL_CASES)
def test_modelspec_wraps_same_functions(model, sim_fn, fit_fn, expected_id):
    """The ModelSpec should bundle the exact same function objects that are
    also exported standalone — not a divergent copy."""
    assert model.sim is sim_fn
    assert model.fit is fit_fn
