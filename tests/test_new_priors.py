import numpy as np
import pytest
from scipy import stats
from pyem.core.priors import (
    GaussianPrior, UniformPrior, LaplacePrior, StudentTPrior,
    CauchyPrior, IndependentPrior, default_prior,
)


def test_gaussian_unchanged():
    g = GaussianPrior(mu=[0.0, 0.0], sigma=[100.0, 100.0])  # sigma == variance
    x = np.array([0.3, -0.2])
    expected = float(-0.5 * np.sum(np.log(2 * np.pi * 100.0) + x ** 2 / 100.0))
    assert np.isclose(g.logpdf(x), expected)


def test_uniform_prior():
    p = UniformPrior(lo=[-1.0, 0.0], hi=[1.0, 2.0])
    assert np.isclose(p.logpdf([0.0, 1.0]), -np.log(2.0) - np.log(2.0))
    assert p.logpdf([2.0, 1.0]) == -np.inf
    mu, var = p.init_moments()
    assert np.allclose(mu, [0.0, 1.0]) and np.all(var > 0)


def test_laplace_matches_scipy():
    p = LaplacePrior(loc=[0.0], scale=[1.5])
    x = np.array([0.7])
    assert np.isclose(p.logpdf(x), stats.laplace.logpdf(x, loc=0.0, scale=1.5).sum())


def test_studentt_matches_scipy():
    p = StudentTPrior(loc=[0.0], scale=[2.0], df=[5.0])
    x = np.array([1.1])
    assert np.isclose(p.logpdf(x), stats.t.logpdf(x, df=5.0, loc=0.0, scale=2.0).sum())


def test_cauchy_matches_scipy():
    p = CauchyPrior(loc=[0.0], scale=[1.0])
    x = np.array([0.4])
    assert np.isclose(p.logpdf(x), stats.cauchy.logpdf(x, loc=0.0, scale=1.0).sum())


def test_independent_sums_children():
    comp = IndependentPrior([GaussianPrior(mu=[0.0], sigma=[1.0]),
                             LaplacePrior(loc=[0.0], scale=[1.0])])
    x = np.array([0.5, -0.3])
    expected = (GaussianPrior(mu=[0.0], sigma=[1.0]).logpdf(x[:1])
                + LaplacePrior(loc=[0.0], scale=[1.0]).logpdf(x[1:]))
    assert np.isclose(comp.logpdf(x), expected)
    mu, var = comp.init_moments()
    assert mu.shape == (2,) and var.shape == (2,)


def test_studentt_init_moments_capped_near_df2():
    for df in (1.0, 2.0, 2.001, 5.0):
        p = StudentTPrior(loc=[0.0], scale=[2.0], df=[df])
        mu, var = p.init_moments()
        assert np.all(np.isfinite(var)) and np.all(var > 0)
        assert np.all(var <= 2.0 ** 2 * 100.0 + 1e-9)   # never exceeds the fallback ceiling


def test_independent_rejects_multidim_child():
    with pytest.raises(ValueError):
        IndependentPrior([GaussianPrior(mu=[0.0, 1.0], sigma=[1.0, 1.0])])


def test_prior_construction_guards():
    with pytest.raises(ValueError):
        UniformPrior(lo=[1.0], hi=[0.0])       # hi<=lo
    with pytest.raises(ValueError):
        LaplacePrior(loc=[0.0], scale=[0.0])   # scale<=0
    with pytest.raises(ValueError):
        StudentTPrior(loc=[0.0], scale=[1.0], df=[0.0])  # df<=0


def test_gaussian_prior_variance_semantics():
    import numpy as np
    from pyem.core.priors import GaussianPrior
    g = GaussianPrior(mu=[0.0], sigma=[4.0])  # sigma is VARIANCE (=4 -> sd 2)
    expected = -0.5 * (np.log(2 * np.pi * 4.0) + 0.0)
    assert np.isclose(g.logpdf([0.0]), expected)
    assert np.allclose(g.variance, g.sigma)
