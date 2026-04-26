"""Shared utilities for generated model files.

Keep this file lightweight and shared across all generated model modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    """Container for one generated model variant."""

    id: str
    spec: dict
    desc: str
    params: Callable
    sim: Callable
    fit: Callable


@dataclass(frozen=True)
class ParamDef:
    """Definition for one parameter in the registry."""

    name: str
    xform: Callable
    init_fn: Callable


def spec_to_id(spec: dict) -> str:
    """Convert a nested spec dictionary into a deterministic ID string."""
    block_order = ["rl", "cr", "link"]
    op_alias = {"linear": "lin"}

    blocks = []
    for block in block_order:
        if block not in spec or not spec[block]:
            continue
        ops = spec[block]
        op_strs = []
        for op_name in sorted(ops.keys()):
            args = ops[op_name]
            name = op_alias.get(op_name, op_name)
            if isinstance(args, dict):
                for subop in sorted(args.keys()):
                    subargs = args[subop]
                    if not isinstance(subargs, (list, tuple)):
                        raise ValueError(
                            f"Arguments for {block}:{op_name}:{subop} must be a list"
                        )
                    op_strs.append(f"{name}.{subop}({','.join(subargs)})")
            elif isinstance(args, (list, tuple)):
                op_strs.append(f"{name}({','.join(args)})")
            else:
                raise ValueError(
                    f"Arguments for {block}:{op_name} must be a list or dict"
                )
        blocks.append(f"{block}={'/'.join(op_strs)}")
    return "|".join(blocks)


def _alloc_sim(nsubj: int, nblocks: int, ntrials: int, nchoices: int = 2) -> Dict[str, np.ndarray]:
    """Allocate common simulation arrays."""
    return {
        "choices": np.zeros((nsubj, nblocks, ntrials), dtype=object),
        "ev": np.zeros((nsubj, nblocks, ntrials + 1, nchoices), dtype=float),
        "ch_prob": np.zeros((nsubj, nblocks, ntrials, nchoices), dtype=float),
        "pe": np.zeros((nsubj, nblocks, ntrials), dtype=float),
        "nll": np.zeros((nsubj,), dtype=float),
    }


def _alloc_fit(nblocks: int, ntrials: int, nchoices: int = 2) -> Dict[str, np.ndarray]:
    """Allocate common fitting arrays."""
    return {
        "ev": np.zeros((nblocks, ntrials + 1, nchoices), dtype=float),
        "ch_prob": np.zeros((nblocks, ntrials, nchoices), dtype=float),
        "pe": np.zeros((nblocks, ntrials), dtype=float),
        "nll": 0.0,
    }


PARAM_REGISTRY = {
    "beta": ParamDef("beta", lambda x: x, lambda rng, n: rng.uniform(0.5, 8.0, size=n)),
    "alpha": ParamDef("alpha", lambda x: x, lambda rng, n: rng.uniform(0.1, 0.9, size=n)),
}


def build_params(
    param_names: Sequence[str],
    nsubj: int,
    rng: np.random.Generator | None = None,
) -> tuple[list[str], list[Callable], np.ndarray]:
    """Build parameter transforms and sampled true params."""
    if rng is None:
        rng = np.random.default_rng()

    true_params = np.zeros((nsubj, len(param_names)), dtype=float)
    param_xform: list[Callable] = []

    for i, name in enumerate(param_names):
        p = PARAM_REGISTRY[name]
        param_xform.append(p.xform)
        true_params[:, i] = p.init_fn(rng, nsubj)

    return list(param_names), param_xform, true_params
