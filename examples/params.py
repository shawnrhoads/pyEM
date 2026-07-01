"""Reference parameter-registry pattern for authoring new pyEM models.

This module is example scaffolding, not a dependency of the ``pyem`` package
itself. The package intentionally has no opinion on what parameters a model
uses, so this registry lives here as a copyable pattern: pick the entries you
need (or add your own), and pass ``param_names`` to :func:`build_params` to
get named/bounded/transformed "true" parameters for simulation.

Each :class:`ParamDef` carries two related-but-distinct ranges:

- ``sample``: how to draw a plausible "true" value for simulation (e.g. a
  learning rate you'd expect a real subject to have).
- ``bounds``: the full valid natural-space range used to reject invalid
  values during fitting (e.g. any learning rate in [0, 1]).

Conflating these two (or hand-copying ``bounds`` into every model's fit
function) is how mismatched bounds creep in across model files.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from pyem.utils.math import norm2alpha, norm2beta


@dataclass(frozen=True)
class ParamDef:
    name: str
    xform: Callable[[np.ndarray], np.ndarray]                    # Gaussian -> natural space
    sample: Callable[[np.random.Generator, int], np.ndarray]      # draw "true" natural-space values
    bounds: tuple[float, float]                                   # valid natural-space range


def _uniform_sampler(lo: float, hi: float) -> Callable[[np.random.Generator, int], np.ndarray]:
    """Build a named, reusable sampler that draws Uniform(lo, hi) values.

    Used instead of an inline ``lambda`` so each sampler is inspectable and
    testable on its own rather than an anonymous closure buried in a dict.
    """
    def sample(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(lo, hi, size=n)
    return sample


PARAM_REGISTRY: dict[str, ParamDef] = {
    ### REINFORCEMENT LEARNING (pyem.models.rl) ###
    "beta":            ParamDef("beta", norm2beta, _uniform_sampler(0.9, 8.5), bounds=(1e-5, 20.0)),
    "alpha":           ParamDef("alpha", norm2alpha, _uniform_sampler(0.05, 0.95), bounds=(0.0, 1.0)),
    "alpha_pos":       ParamDef("alpha_pos", norm2alpha, _uniform_sampler(0.05, 0.95), bounds=(0.0, 1.0)),
    "alpha_neg":       ParamDef("alpha_neg", norm2alpha, _uniform_sampler(0.05, 0.95), bounds=(0.0, 1.0)),
    "alpha_self":      ParamDef("alpha_self", norm2alpha, _uniform_sampler(0.05, 0.95), bounds=(0.0, 1.0)),
    "alpha_other":     ParamDef("alpha_other", norm2alpha, _uniform_sampler(0.05, 0.95), bounds=(0.0, 1.0)),
    "alpha_noone":     ParamDef("alpha_noone", norm2alpha, _uniform_sampler(0.05, 0.95), bounds=(0.0, 1.0)),
    "alpha_self_pos":  ParamDef("alpha_self_pos", norm2alpha, _uniform_sampler(0.05, 0.95), bounds=(0.0, 1.0)),
    "alpha_self_neg":  ParamDef("alpha_self_neg", norm2alpha, _uniform_sampler(0.05, 0.95), bounds=(0.0, 1.0)),
    "alpha_other_pos": ParamDef("alpha_other_pos", norm2alpha, _uniform_sampler(0.05, 0.95), bounds=(0.0, 1.0)),
    "alpha_other_neg": ParamDef("alpha_other_neg", norm2alpha, _uniform_sampler(0.05, 0.95), bounds=(0.0, 1.0)),

    ### BAYESIAN BELIEF UPDATING (pyem.models.bayes) ###
    "lambda1":         ParamDef("lambda1", norm2alpha, _uniform_sampler(0.2, 0.8), bounds=(0.0, 1.0)),

    ### LINEAR MODELS (pyem.models.glm) ###
    "gamma":           ParamDef("gamma", norm2alpha, _uniform_sampler(0.05, 0.95), bounds=(0.0, 1.0)),
    "phi":             ParamDef("phi", norm2alpha, _uniform_sampler(-0.9, 0.9), bounds=(-0.999, 0.999)),
}


def build_params(
    param_names: list[str],
    nsubj: int,
    rng: np.random.Generator | None = None,
) -> tuple[list[str], list[Callable], np.ndarray]:
    """Draw "true" natural-space parameters for ``nsubj`` simulated subjects.

    Returns ``(param_names, param_xform, true_params)`` where ``true_params``
    has shape ``(nsubj, len(param_names))``.
    """
    if rng is None:
        rng = np.random.default_rng()

    true_params = np.zeros((nsubj, len(param_names)))
    param_xform = []

    for i, name in enumerate(param_names):
        p = PARAM_REGISTRY[name]
        param_xform.append(p.xform)
        true_params[:, i] = p.sample(rng, nsubj)

    return param_names, param_xform, true_params


def validate_params(param_names: list[str], natural_values) -> float | None:
    """Return a large penalty if any natural-space value is out of its
    registered ``bounds``, else ``None``.

    Example usage inside a hand-written ``*_fit`` function:

        beta = norm2beta(params[0]); alpha = norm2alpha(params[1])
        penalty = validate_params(["beta", "alpha"], [beta, alpha])
        if penalty is not None:
            return penalty
    """
    for name, val in zip(param_names, natural_values):
        lo, hi = PARAM_REGISTRY[name].bounds
        if not (lo <= val <= hi):
            return 1e7
    return None
