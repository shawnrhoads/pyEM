import numpy as np
from pyem.models.discounting import (
    sd_hyp_k_sim, sd_hyp_k_fit,
    td_hyp_k_sim, td_hyp_k_fit,
    ped_par_2k_sim, ped_par_2k_fit,
)


def test_sd_hyp_k_roundtrip():
    out = sd_hyp_k_sim(np.array([[0.1]]), seed=0)
    nll = sd_hyp_k_fit(np.array([0.0]), out["choices"][0], out["payouts"][0],
                       out["social_dists"][0], output="nll")
    assert np.isfinite(nll)


def test_td_hyp_k_roundtrip():
    out = td_hyp_k_sim(np.array([[0.05]]), seed=0)
    nll = td_hyp_k_fit(np.array([0.0]), out["choices"][0], out["payouts"][0],
                       out["delays"][0], output="nll")
    assert np.isfinite(nll)


def test_ped_par_2k_roundtrip():
    out = ped_par_2k_sim(np.array([[0.5, 0.8]]), seed=0)
    nll = ped_par_2k_fit(np.array([0.0, 0.0]), out["choices"][0], out["payouts"][0],
                         out["effort_levels"][0], out["beneficiary"][0], output="nll")
    assert np.isfinite(nll)
