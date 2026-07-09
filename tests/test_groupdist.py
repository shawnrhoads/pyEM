import numpy as np
import pytest
from pyem.core.groupdist import make_group, GaussianGroup


def _ref_gaussian(m, inv_h):
    nsub = m.shape[1]; npar = m.shape[0]
    mu = np.mean(m, axis=1)
    sigma = np.zeros(npar)
    for s in range(nsub):
        sigma += m[:, s] ** 2 + np.diag(inv_h[:, :, s])
    sigma = sigma / nsub - mu ** 2
    return mu, sigma, int(np.min(sigma) >= 0)


def test_gaussian_matches_reference():
    rng = np.random.default_rng(0)
    m = rng.standard_normal((3, 20))
    inv_h = np.stack([np.eye(3) * 0.1 for _ in range(20)], axis=-1)
    mu, sigma, flag = GaussianGroup().update(m, inv_h)
    rmu, rsigma, rflag = _ref_gaussian(m, inv_h)
    assert np.allclose(mu, rmu) and np.allclose(sigma, rsigma) and flag == rflag


def test_make_group_dispatch():
    assert make_group("cauchy").df == 1.0
    for name in ("gaussian", "laplace", "student_t", "cauchy"):
        g = make_group(name)
        assert hasattr(g, "update") and hasattr(g, "make_prior") and hasattr(g, "moments")


def test_laplace_recovers_location_and_scale():
    rng = np.random.default_rng(1)
    true_loc, true_b = 0.5, 1.0
    m = (true_loc + rng.laplace(0.0, true_b, size=(1, 400)))
    inv_h = np.stack([np.eye(1) * 1e-6 for _ in range(400)], axis=-1)
    g = make_group("laplace")
    hyper = g.update(m, inv_h)
    assert abs(hyper["loc"][0] - true_loc) < 0.15
    assert 1 / 1.5 < hyper["scale"][0] / true_b < 1.5


def test_studentt_recovers_location():
    rng = np.random.default_rng(2)
    true_loc = -0.3
    m = (true_loc + rng.standard_t(5, size=(1, 400)))
    inv_h = np.stack([np.eye(1) * 1e-6 for _ in range(400)], axis=-1)
    g = make_group("student_t", df=5.0)
    hyper = g.update(m, inv_h)
    assert abs(hyper["loc"][0] - true_loc) < 0.2


def test_make_prior_finite_at_loc():
    for name in ("gaussian", "laplace", "student_t", "cauchy"):
        g = make_group(name)
        m = np.zeros((2, 30)); inv_h = np.stack([np.eye(2) for _ in range(30)], axis=-1)
        hyper = g.update(m, inv_h)
        prior = g.make_prior(hyper)
        assert np.isfinite(prior.logpdf(np.zeros(2)))


def test_make_group_unknown_raises():
    with pytest.raises(ValueError):
        make_group("bogus")


def test_groupdist_single_subject_and_single_param():
    import numpy as np
    for name in ("gaussian", "laplace", "student_t", "cauchy"):
        g = make_group(name)
        # single subject, 2 params
        m1 = np.zeros((2, 1)); ih1 = np.eye(2).reshape(2, 2, 1)
        mu, sigma, flag = g.moments(g.update(m1, ih1))
        assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(sigma > 0)
        # single parameter, several subjects
        m2 = np.zeros((1, 5)); ih2 = np.stack([np.eye(1) for _ in range(5)], axis=-1)
        mu, sigma, flag = g.moments(g.update(m2, ih2))
        assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(sigma > 0)
