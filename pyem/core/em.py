from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Sequence, Any, Literal
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from .priors import GaussianPrior, default_prior
from .optim import OptimConfig, single_subject_minimize

ConvergenceMethod = Literal["sum","mean","median"]
ConvergenceType = Literal["NPL","LME"]

@dataclass
class EMConfig:
    mstep_maxit: int = 200
    estep_maxit: int | None = None            # kept for compatibility; we restart via optimizer
    convergence_method: ConvergenceMethod = "sum"
    convergence_type: ConvergenceType = "NPL"
    convergence_custom: Literal["relative_npl","running_average", None] = None
    convergence_crit: float = 1e-3
    convergence_precision: int = 6
    njobs: int = -1
    optim: OptimConfig = field(default_factory=OptimConfig)
    seed: int | None = None
    max_subject_retries: int = 0              # additional retries if optimizer fails badly
    compute_lme: bool = False                 # compute Laplace approximation

def _hier_convergence(vals: np.ndarray, method: ConvergenceMethod) -> float:
    if method == "sum":
        return float(np.sum(vals))
    elif method == "mean":
        return float(np.mean(vals))
    else:
        return float(np.median(vals))

def _calc_group_gaussian(m: np.ndarray, inv_h: np.ndarray, covmat: bool = False) -> tuple[np.ndarray,np.ndarray,int]:
    # m: (nparams, nsubjects), inv_h: (nparams,nparams,nsubjects)
    nsub = m.shape[1]
    npar = m.shape[0]
    mu = np.mean(m, axis=1)
    sigma = np.zeros(npar)
    for s in range(nsub):
        sigma += m[:, s]**2 + np.diag(inv_h[:, :, s])
    sigma = sigma / nsub - mu**2
    flag = 1
    if np.min(sigma) < 0:
        flag = 0

    if not covmat:
        return mu, sigma, flag
    else:
        covmat = np.zeros((npar, npar))
        for isub in range(nsub):
            covmat += np.outer(m[:, isub], m[:, isub]) - np.outer(m[:, isub], mu) - np.outer(mu, m[:, isub]) + np.outer(mu, mu) + inv_h[:, :, isub]
        covmat /= nsub

        if np.linalg.det(covmat) <= 0:
            print('Negative/zero determinant - prior covariance not updated')

        return mu, sigma, flag, covmat

def EMfit(
    all_data: Sequence[Sequence[Any]] | Sequence[pd.DataFrame],
    objfunc: Callable[..., float],
    param_names: Sequence[str],
    *,
    verbose: int = 1,
    config: EMConfig | None = None,
    prior: GaussianPrior | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Hierarchical EM with MAP. Compatible with legacy signature.

    all_data: list where item i contains args for objfunc for subject i
              (e.g., [choices, rewards]) or a pandas DataFrame for subject i.
    objfunc: callable like f(params, *subject_args, prior=None, output='npl') -> float
    param_names: list of parameter names (nparams)
    """
    nsubjects = len(all_data)
    nparams = len(param_names)
    if config is None:
        config = EMConfig()
    # Backward-compat kwargs mapping (e.g., mstep_maxit=..., njobs=..., etc.)
    if kwargs:
        if 'mstep_maxit' in kwargs: config.mstep_maxit = int(kwargs['mstep_maxit'])
        if 'estep_maxit' in kwargs: config.estep_maxit = kwargs['estep_maxit']
        if 'convergence_method' in kwargs: config.convergence_method = kwargs['convergence_method']
        if 'convergence_type' in kwargs: config.convergence_type = kwargs['convergence_type']
        if 'convergence_custom' in kwargs: config.convergence_custom = kwargs['convergence_custom']
        if 'convergence_crit' in kwargs: config.convergence_crit = float(kwargs['convergence_crit'])
        if 'convergence_precision' in kwargs: config.convergence_precision = int(kwargs['convergence_precision'])
        if 'njobs' in kwargs: config.njobs = int(kwargs['njobs'])
        if 'seed' in kwargs: config.seed = kwargs['seed']
        # Optimizer knobs
        if 'optim_method' in kwargs or 'optim_options' in kwargs or 'max_restarts' in kwargs:
            from .optim import OptimConfig
            config.optim = OptimConfig(
                method=kwargs.get('optim_method', config.optim.method),
                options=kwargs.get('optim_options', config.optim.options),
                max_restarts=kwargs.get('max_restarts', config.optim.max_restarts)
            )
    rng = np.random.default_rng(config.seed)
    if prior is None:
        prior = default_prior(nparams, seed=config.seed)

    # initialize tracking
    NPL_hist = []
    NPL_old = np.inf
    converged = False

    # initial posterior
    post_mu = prior.mu.copy()
    post_sigma = prior.sigma.copy()

    for iiter in range(config.mstep_maxit):
        # build per-iteration prior object (closure for current posterior)
        iter_prior = GaussianPrior(mu=post_mu.copy(), sigma=post_sigma.copy())

        # E-step in parallel
        def _fit_subject(i: int):
            args = all_data[i]
            # normalize args into tuple for *args
            if isinstance(args, pd.DataFrame):
                obj_args = (args,)
            else:
                obj_args = tuple(args)
            return single_subject_minimize(
                objfunc=objfunc,
                obj_args=obj_args,
                nparams=nparams,
                prior=iter_prior,
                config=config.optim,
                rng=np.random.default_rng((config.seed or 0) + i)
            )

        results = Parallel(n_jobs=config.njobs)(delayed(_fit_subject)(i) for i in range(nsubjects))

        m = np.zeros((nparams, nsubjects))
        inv_h = np.zeros((nparams, nparams, nsubjects))
        NPL = np.zeros(nsubjects)
        NLP = np.zeros(nsubjects)

        for i, (q_est, hess_inv, fval, nl_prior, success, res) in enumerate(results):
            m[:, i] = q_est
            inv_h[:, :, i] = hess_inv
            NPL[i] = round(fval, config.convergence_precision)
            NLP[i] = nl_prior

        # M-step: empirical Bayes update (group Gaussian)
        mu, sigma, ok = _calc_group_gaussian(m, inv_h)
        if ok:
            post_mu = mu
            post_sigma = sigma

        conv_val = _hier_convergence(NPL, config.convergence_method)
        if verbose:
            if (iiter == 0) or conv_val <= min(NPL_hist + [conv_val]):
                if config.convergence_custom == "relative_npl" and iiter > 0:
                    print(f"{abs((conv_val - NPL_old) / (NPL_old if NPL_old != 0 else 1.0)):.4f} ({iiter:03d})", end=", ")
                else:
                    print(f"{conv_val:.4f} ({iiter:03d})", end=", ")

        # convergence check
        if config.convergence_custom == "running_average" and ok and iiter > 5:
            if abs(conv_val - float(np.mean(NPL_hist[-5:]))) < config.convergence_crit:
                converged = True
        elif config.convergence_custom == "relative_npl" and ok and iiter > 1:
            if abs((conv_val - NPL_old) / (NPL_old if NPL_old != 0 else 1.0)) < config.convergence_crit:
                converged = True
        else:
            if iiter > 0 and abs(conv_val - NPL_old) < config.convergence_crit and ok:
                converged = True

        NPL_hist.append(conv_val)
        NPL_old = conv_val

        if converged:
            break

    out = {
        "m": m,
        "inv_h": inv_h,
        "posterior": {"mu": post_mu, "sigma": post_sigma},
        "NPL": NPL,
        "NLPrior": NLP,
        "NLL": NPL - NLP,
        "convergence": converged,
    }
    return out
