
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Any, Optional
import numpy as np
from scipy.optimize import minimize, OptimizeResult

ObjectiveFn = Callable[..., float]

@dataclass
class OptimConfig:
    method: str = "BFGS"                 # full inverse Hessian available
    options: dict | None = None
    max_restarts: int = 2                # extra random initializations if not successful
    tol: float = 1e-6                    # fun convergence tolerance
    x_scale: float = 0.1                 # scale of random initializations

def single_subject_minimize(
    objfunc: ObjectiveFn,
    obj_args: Iterable[Any],
    nparams: int,
    prior: Any,
    config: OptimConfig,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, float, float, bool, OptimizeResult]:
    """
    Returns (q_est, inv_h, fval, nl_prior, success, result)
    """
    options = {"gtol": 1e-4, "eps": 1e-4}
    if config.options:
        options.update(config.options)

    best = None
    best_result: Optional[OptimizeResult] = None
    for attempt in range(1 + config.max_restarts):
        x0 = config.x_scale * rng.standard_normal(nparams)
        result = minimize(
            lambda x, *args: objfunc(x, *args, prior),
            x0=x0,
            args=tuple(obj_args),
            method=config.method,
            options=options
        )
        if best is None or result.fun < best:
            best = float(result.fun)
            best_result = result
        if result.success:
            break  # accept successful result

    assert best_result is not None  # for type checker

    q_est = best_result.x
    fval = float(best_result.fun)
    # Hessian inverse (BFGS provides it). Fallback: diagonal approx.
    if hasattr(best_result, "hess_inv"):
        h = best_result.hess_inv
        try:
            inv_h = np.asarray(h)
        except Exception:
            n = nparams
            inv_h = np.eye(n) * max(1.0, np.linalg.norm(q_est) + 1e-6)
    else:
        n = nparams
        inv_h = np.eye(n) * max(1.0, np.linalg.norm(q_est) + 1e-6)

    nl_prior = -prior.logpdf(q_est)
    success = bool(best_result.success)
    return q_est, inv_h, fval, nl_prior, success, best_result
