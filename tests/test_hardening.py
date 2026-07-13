import numpy as np
from pyem.utils.math import calc_fval
from pyem.api import EMModel
from pyem.models.rl_mf import rw1a1b_fit, rw1a1b_sim
from test_helpers import _simulate_rw_params


def test_calc_fval_nonfinite_nll():
    assert calc_fval(np.nan, np.array([0.0, 0.0])) == 1e7
    assert calc_fval(np.inf, np.array([0.0, 0.0])) == 1e7
    assert calc_fval(np.nan, np.array([0.0, 0.0]), output="nll") == 1e7

    # Pin the npl+prior branch specifically (a finite log-prior + NaN NLL must
    # still cap to 1e7 via `not np.isfinite`, which the old `np.isinf`-only
    # guard missed). Without a prior, calc_fval short-circuits to the else branch.
    class _FinitePrior:
        def logpdf(self, x):
            return 0.0
    assert calc_fval(np.nan, np.array([0.0, 0.0]), prior=_FinitePrior(), output="npl") == 1e7


def test_single_subject_fit():
    params = _simulate_rw_params(1)
    sim = rw1a1b_sim(params, nblocks=1, ntrials=6)
    all_data = [[sim["choices"][0], sim["rewards"][0]]]
    res = EMModel(all_data, rw1a1b_fit, ["beta", "alpha"]).fit(verbose=0, seed=0, mstep_maxit=5, njobs=1)
    assert res.m.shape == (2, 1) and np.all(np.isfinite(res.m))


def test_single_parameter_model():
    def one_p_fit(params, x, prior=None, output="npl"):
        from pyem.utils.math import calc_fval
        nll = float(np.sum((np.asarray(x) - params[0]) ** 2))
        if output == "all":
            return {"params": [params[0]], "nll": nll}
        return calc_fval(nll, params, prior=prior, output=output)
    all_data = [[np.array([0.1, 0.2, 0.3])] for _ in range(5)]
    res = EMModel(all_data, one_p_fit, ["theta"]).fit(verbose=0, seed=0, mstep_maxit=5, njobs=1)
    assert res.m.shape == (1, 5) and np.all(np.isfinite(res.m))


def test_nan_subject_does_not_poison_fit():
    # a fit func that returns NaN NLL for one subject; calc_fval must cap it -> fit still finite
    calls = {"n": 0}
    def flaky_fit(params, x, prior=None, output="npl"):
        from pyem.utils.math import calc_fval
        nll = float(np.sum((np.asarray(x) - params[0]) ** 2))
        if np.asarray(x)[0] < 0:   # designate "bad" subjects by data sign
            nll = np.nan
        if output == "all":
            return {"params": [params[0]], "nll": nll if np.isfinite(nll) else 1e7}
        return calc_fval(nll, params, prior=prior, output=output)
    all_data = [[np.array([0.5, 0.5])] for _ in range(4)] + [[np.array([-1.0, -1.0])]]
    res = EMModel(all_data, flaky_fit, ["theta"]).fit(verbose=0, seed=0, mstep_maxit=5, njobs=1)
    assert np.all(np.isfinite(res.m))


def test_empty_all_data_raises():
    import pytest
    with pytest.raises(ValueError):
        EMModel([], rw1a1b_fit, ["beta", "alpha"]).fit(verbose=0, njobs=1)


def test_empty_subject_entry_raises():
    import pytest
    all_data = [[np.array([0.1, 0.2])], []]
    with pytest.raises(ValueError):
        EMModel(all_data, rw1a1b_fit, ["beta", "alpha"]).fit(verbose=0, njobs=1)


def test_empty_param_names_raises():
    import pytest
    params = _simulate_rw_params(3)
    sim = rw1a1b_sim(params, nblocks=1, ntrials=6)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
    with pytest.raises(ValueError):
        EMModel(all_data, rw1a1b_fit, []).fit(verbose=0, njobs=1)


def test_emfit_rejects_zero_maxit():
    import pytest
    from pyem.core.em import EMfit
    sim = rw1a1b_sim(np.array([[3.0, 0.5]]), nblocks=1, ntrials=5, seed=0)
    all_data = [[sim["choices"][0], sim["rewards"][0]]]
    with pytest.raises(ValueError):
        EMfit(all_data, rw1a1b_fit, ["beta", "alpha"], verbose=0, mstep_maxit=0)
