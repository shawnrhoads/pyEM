import numpy as np
from pyem.models.bayes import bayes_sim, bayes_fit


def test_bayes_fit_infers_n_fish():
    true = np.array([[0.6]])
    sim = bayes_sim(true, nblocks=3, ntrials=8, n_fish=3, seed=0)
    nll = bayes_fit(np.array([0.0]), sim["choices"][0], sim["observations"][0],
                    n_fish=3, output="nll")
    assert np.isfinite(nll)


def test_bayes_fit_default_infers_from_data():
    true = np.array([[0.6]])
    sim = bayes_sim(true, nblocks=3, ntrials=8, n_fish=3, seed=1)
    # no n_fish passed -> inferred from data, must run without shape error
    nll = bayes_fit(np.array([0.0]), sim["choices"][0], sim["observations"][0],
                    output="nll")
    assert np.isfinite(nll)


def test_bayes_sim_seed_reproducible():
    true = np.array([[0.6]])
    a = bayes_sim(true, nblocks=2, ntrials=6, seed=3)
    b = bayes_sim(true, nblocks=2, ntrials=6, seed=3)
    assert np.array_equal(a["choices"], b["choices"])
    c = bayes_sim(true, nblocks=2, ntrials=6, seed=4)
    assert not np.array_equal(a["choices"], c["choices"])
