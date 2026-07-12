import numpy as np
from pyem.models.rl_mf import rw4a1b_sim, rw4a1b_fit, rw1a1b_sim


def test_rw4a1b_fit_roundtrip_finite():
    """H2 regression: rw4a1b_fit runs and produces a finite NLL."""
    rng = np.random.default_rng(0)
    beta = rng.uniform(1.0, 5.0, size=2)
    alphas = rng.uniform(0.1, 0.9, size=(2, 4))
    true = np.column_stack([beta, alphas])
    sim = rw4a1b_sim(true, nblocks=6, ntrials=20, seed=0)
    params = np.zeros(5)
    out = rw4a1b_fit(params,
                     sim["choices"][0], sim["outcomes_self"][0],
                     sim["outcomes_other"][0], sim["option_pairs"][0],
                     output="all")
    assert np.isfinite(out["nll"])


def test_rw_sim_seed_reproducible():
    """L8: same seed -> identical simulated choices."""
    true = np.array([[3.0, 0.5]])
    a = rw1a1b_sim(true, nblocks=2, ntrials=10, seed=7)
    b = rw1a1b_sim(true, nblocks=2, ntrials=10, seed=7)
    assert np.array_equal(a["choices"], b["choices"])


def test_rw4a1b_sim_seed_reproducible():
    """L8: rw4a1b_sim fully seeded (blocks + outcomes) -> identical output."""
    true = np.array([[3.0, 0.5, 0.5, 0.5, 0.5]])
    a = rw4a1b_sim(true, nblocks=6, ntrials=20, seed=11)
    b = rw4a1b_sim(true, nblocks=6, ntrials=20, seed=11)
    assert np.array_equal(a["choices"], b["choices"])
    assert np.array_equal(a["option_pairs"], b["option_pairs"])
    assert np.array_equal(a["outcomes_self"], b["outcomes_self"])
