import numpy as np
from pyem.api import EMModel
from pyem.core.priors import UniformPrior
from pyem.models.rl_mf import rw1a1b_fit as rw_fit, rw1a1b_sim as rw_sim
from test_helpers import _simulate_rw_params

def test_uniform_prior_and_fit():
    """Fit with a real UniformPrior and check the estimates respect its support.

    NOTE: despite its name, this test previously exercised GaussianPrior, not
    UniformPrior at all -- a mislabeled leftover. UniformPrior's own logpdf /
    out-of-bounds / init_moments behavior is already covered directly by
    tests/test_new_priors.py::test_uniform_prior, and a fit-level smoke check
    (UniformPrior + EMModel.fit -> finite estimates) is already covered by
    tests/test_fitting.py::test_uniform_prior_convergent_fit. Rather than drop
    this test as a pure duplicate, it is rewritten here to actually use
    UniformPrior and to assert something neither of those tests check: that
    the fitted subject-level estimates stay within the prior's [lo, hi] box.
    """
    nsubjects, nblocks, ntrials = 10, 1, 4
    params = _simulate_rw_params(nsubjects)
    sim = rw_sim(params, nblocks=nblocks, ntrials=ntrials)
    all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]

    lo, hi = [-5.0, -5.0], [5.0, 5.0]
    prior = UniformPrior(lo=lo, hi=hi)
    model = EMModel(all_data=all_data, fit_func=rw_fit, param_names=["beta", "alpha"])
    # Only iteration 0 uses the UniformPrior directly (later iterations use the
    # Gaussian empirical-Bayes prior, which does not enforce the box), so run a
    # single iteration to make the support-containment assertion valid by construction.
    result = model.fit(prior=prior, verbose=0, seed=0, mstep_maxit=1, njobs=1)

    assert result.m.shape == (2, nsubjects)
    assert np.all(np.isfinite(result.m))
    assert np.all(result.m >= np.array(lo)[:, None] - 1e-6)
    assert np.all(result.m <= np.array(hi)[:, None] + 1e-6)

