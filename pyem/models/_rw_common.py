"""Shared output-array allocation for the 2-option RW/softmax family (rw1a1b, rw2a1b).

Field names match exactly what ``rl.py`` has always returned — this is a
straight extraction of the previously-inline allocation code, not a redesign
of the output contract. ``rw3a1b``/``rw4a1b`` have incompatible shapes (extra
outcome channels, 4-option pairs) and keep their own bespoke allocation.
"""
from __future__ import annotations
import numpy as np
from typing import Dict


def _alloc_rw_sim(nsubj: int, nblocks: int, ntrials: int) -> Dict[str, np.ndarray]:
    return dict(
        choices   = np.empty((nsubj, nblocks, ntrials), dtype=object),        # actual choice ('A'/'B')
        rewards   = np.zeros((nsubj, nblocks, ntrials), dtype=float),         # actual reward
        EV        = np.zeros((nsubj, nblocks, ntrials + 1, 2), dtype=float),  # expected values
        ch_prob   = np.zeros((nsubj, nblocks, ntrials, 2), dtype=float),      # choice probabilities
        choices_A = np.zeros((nsubj, nblocks, ntrials), dtype=float),        # choice A binary
        PE        = np.zeros((nsubj, nblocks, ntrials), dtype=float),         # prediction errors
        nll       = np.zeros((nsubj, nblocks, ntrials), dtype=float),         # negative log-likelihood
    )


def _alloc_rw_fit(nblocks: int, ntrials: int) -> Dict[str, np.ndarray | float]:
    return dict(
        EV  = np.zeros((nblocks, ntrials + 1, 2), dtype=float),  # expected values
        PE  = np.zeros((nblocks, ntrials), dtype=float),          # prediction errors
        nll = 0.0,                                                # negative log-likelihood accumulator
    )
