from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Sequence, Any, Literal
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from .priors import Prior, default_prior
from .optim import OptimConfig, single_subject_minimize
from .groupdist import make_group

ConvergenceMethod = Literal["sum","mean","median"]

@dataclass
class EMConfig:
    mstep_maxit: int = 200
    convergence_method: ConvergenceMethod = "sum"
    convergence_custom: Literal["relative_npl","running_average", None] = None
    convergence_crit: float = 1e-3
    convergence_precision: int = 6
    njobs: int = -2
    optim: OptimConfig = field(default_factory=OptimConfig)
    seed: int | None = None
    mstep: str = "gaussian"

def _hier_convergence(vals: np.ndarray, method: ConvergenceMethod) -> float:
    if method == "sum":
        return float(np.sum(vals))
    elif method == "mean":
        return float(np.mean(vals))
    else:
        return float(np.median(vals))

def EMfit(
    all_data: Sequence[Sequence[Any]] | Sequence[pd.DataFrame],
    objfunc: Callable[..., float],
    param_names: Sequence[str],
    *,
    verbose: int = 1,
    config: EMConfig | None = None,
    prior: Prior | None = None,
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
    if nsubjects == 0:
        raise ValueError("EMfit received empty `all_data`: no subjects to fit.")
    if nparams == 0:
        raise ValueError("EMfit received empty `param_names`: at least one parameter is required.")
    for i, _args in enumerate(all_data):
        _empty = _args.empty if isinstance(_args, pd.DataFrame) else (len(_args) == 0)
        if _empty:
            raise ValueError(
                f"all_data[{i}] is empty; each subject's entry must contain non-empty "
                "arguments/data for objfunc."
            )
    if config is None:
        config = EMConfig()
    # Backward-compat kwargs mapping (e.g., mstep_maxit=..., njobs=..., etc.)
    if kwargs:
        if 'mstep_maxit' in kwargs: config.mstep_maxit = int(kwargs['mstep_maxit'])
        if 'convergence_method' in kwargs: config.convergence_method = kwargs['convergence_method']
        if 'convergence_custom' in kwargs: config.convergence_custom = kwargs['convergence_custom']
        if 'convergence_crit' in kwargs: config.convergence_crit = float(kwargs['convergence_crit'])
        if 'convergence_precision' in kwargs: config.convergence_precision = int(kwargs['convergence_precision'])
        if 'njobs' in kwargs: config.njobs = int(kwargs['njobs'])
        if 'seed' in kwargs: config.seed = kwargs['seed']
        if 'mstep' in kwargs: config.mstep = kwargs['mstep']
        # Optimizer knobs
        if 'optim_method' in kwargs or 'optim_options' in kwargs or 'max_restarts' in kwargs:
            from .optim import OptimConfig
            config.optim = OptimConfig(
                method=kwargs.get('optim_method', config.optim.method),
                options=kwargs.get('optim_options', config.optim.options),
                max_restarts=kwargs.get('max_restarts', config.optim.max_restarts)
            )
    if config.mstep_maxit < 1:
        raise ValueError("mstep_maxit must be >= 1")
    if prior is None:
        prior = default_prior(nparams, seed=config.seed)

    group = make_group(config.mstep)

    # initialize tracking
    NPL_hist = []
    NPL_old = np.inf
    converged = False

    # initial posterior (seeded from the user prior's own moments when available)
    if hasattr(prior, "init_moments"):
        pm, ps = prior.init_moments()
        post_mu, post_sigma = np.asarray(pm, float).copy(), np.asarray(ps, float).copy()
    elif hasattr(prior, "mu") and hasattr(prior, "sigma"):
        post_mu, post_sigma = np.asarray(prior.mu, float).copy(), np.asarray(prior.sigma, float).copy()
    else:
        raise TypeError("prior must define init_moments() or expose .mu/.sigma (GaussianPrior-like)")
    last_good_hyper = None

    with Parallel(n_jobs=config.njobs) as parallel:
        for iiter in range(config.mstep_maxit):
            # Use the user-supplied prior until a valid (ok) M-step has occurred;
            # afterwards use the chosen group family's prior built from the last
            # known-good M-step. This mirrors the pre-refactor semantics, which only
            # ever fed a validated prior into the next E-step.
            iter_prior = prior if last_good_hyper is None else group.make_prior(last_good_hyper)

            # E-step in parallel
            def _fit_subject(i: int, iter_prior=iter_prior):
                args = all_data[i]
                # normalize args into tuple for *args
                if isinstance(args, pd.DataFrame):
                    obj_args = (args,)
                else:
                    obj_args = tuple(args)
                subject_seed = None if config.seed is None else config.seed + i
                return single_subject_minimize(
                    objfunc=objfunc,
                    obj_args=obj_args,
                    nparams=nparams,
                    prior=iter_prior,
                    config=config.optim,
                    rng=np.random.default_rng(subject_seed)
                )

            results = parallel(delayed(_fit_subject)(i) for i in range(nsubjects))

            m = np.zeros((nparams, nsubjects))
            inv_h = np.zeros((nparams, nparams, nsubjects))
            NPL = np.zeros(nsubjects)
            NLP = np.zeros(nsubjects)

            for i, (q_est, hess_inv, fval, nl_prior, success, res) in enumerate(results):
                m[:, i] = q_est
                inv_h[:, :, i] = hess_inv
                NPL[i] = round(fval, config.convergence_precision)
                NLP[i] = nl_prior

            # M-step: empirical Bayes update via the chosen group-distribution family
            hyper = group.update(m, inv_h)
            mu, sigma, ok = group.moments(hyper)
            if ok:
                post_mu = mu
                post_sigma = sigma
                last_good_hyper = hyper
            # if not ok: keep last_good_hyper and post_mu/post_sigma from the last good step

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
