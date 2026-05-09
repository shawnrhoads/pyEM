"""Template for one generated model module."""

from __future__ import annotations

import numpy as np

from pyem.utils.math import calc_fval, norm2alpha, norm2beta, softmax
from modclass_utils import (
    ModelSpec,
    _alloc_fit,
    _alloc_sim,
    build_params,
    spec_to_id,
)


mod_desc = """Replace with concise model description."""
mod_spec = {"rl": {"softmax": ["beta"], "rw": ["alpha"]}}
mod_id = spec_to_id(mod_spec)


def mod_params(nsubj: int, rng: np.random.Generator | None = None):
    """Generate parameter names, transforms, and true parameters."""
    return build_params(["beta", "alpha"], nsubj, rng)


def mod_sim(params: np.ndarray, nblocks: int = 4, ntrials: int = 12, **kwargs):
    """Simulate behavior for this model variant."""
    nsubj = params.shape[0]
    dat = _alloc_sim(nsubj, nblocks, ntrials, nchoices=2)
    rng = np.random.default_rng(kwargs.get("seed", None))

    beta = params[:, 0]
    alpha = params[:, 1]

    for s in range(nsubj):
        for b in range(nblocks):
            dat["ev"][s, b, 0, :] = 0.5
            for t in range(ntrials):
                p = softmax(dat["ev"][s, b, t, :], beta[s])
                c = rng.choice([0, 1], p=p)
                r = float(rng.integers(0, 2))
                dat["choices"][s, b, t] = "A" if c == 0 else "B"
                dat["rewards"][s, b, t] = r
                dat["ch_prob"][s, b, t, :] = p
                dat["pe"][s, b, t] = r - dat["ev"][s, b, t, c]
                dat["ev"][s, b, t + 1, :] = dat["ev"][s, b, t, :]
                dat["ev"][s, b, t + 1, c] = (
                    dat["ev"][s, b, t, c] + alpha[s] * dat["pe"][s, b, t]
                )
                dat["nll"][s] += -np.log(p[c] + 1e-12)

    dat["params"] = params
    return dat


def mod_fit(params, choices, rewards, prior=None, output="npl"):
    """Fit objective (npl/nll) with optional diagnostics."""
    beta = float(norm2beta(params[0]))
    alpha = float(norm2alpha(params[1]))

    if not (1e-5 <= beta <= 20.0) or not (0.0 <= alpha <= 1.0):
        return 1e7

    nblocks, ntrials = rewards.shape
    dat = _alloc_fit(nblocks, ntrials, nchoices=2)

    for b in range(nblocks):
        dat["ev"][b, 0, :] = 0.5
        for t in range(ntrials):
            c = 0 if choices[b, t] == "A" else 1
            p = softmax(dat["ev"][b, t, :], beta)
            r = rewards[b, t]
            dat["ch_prob"][b, t, :] = p
            dat["pe"][b, t] = r - dat["ev"][b, t, c]
            dat["ev"][b, t + 1, :] = dat["ev"][b, t, :]
            dat["ev"][b, t + 1, c] = dat["ev"][b, t, c] + alpha * dat["pe"][b, t]
            dat["nll"] += -np.log(p[c] + 1e-12)

    if output == "all":
        return {"params": [beta, alpha], **dat}
    return calc_fval(dat["nll"], np.asarray(params), prior=prior, output=output)


MODEL = ModelSpec(
    id=mod_id,
    spec=mod_spec,
    desc=mod_desc,
    params=mod_params,
    sim=mod_sim,
    fit=mod_fit,
)
