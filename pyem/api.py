
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Sequence
import numpy as np
from .core.em import EMfit, EMConfig
from .core.priors import GaussianPrior

@dataclass
class FitResult:
    m: np.ndarray
    inv_h: np.ndarray
    posterior_mu: np.ndarray
    posterior_sigma: np.ndarray
    NPL: np.ndarray
    NLPrior: np.ndarray
    NLL: np.ndarray
    convergence: bool

class EMModel:
    """
    A high-level, sklearn-like interface:
      model = EMModel(all_data, fit_func, param_names, simulate_func=...)
      result = model.fit(...)
      sim = model.simulate(...)
    """
    def __init__(
        self,
        all_data: Sequence[Sequence[Any]] | None,
        fit_func: Callable[..., float],
        param_names: Sequence[str],
        simulate_func: Callable[..., Any] | None = None,
    ) -> None:
        self.all_data = all_data
        self.fit_func = fit_func
        self.param_names = list(param_names)
        self.simulate_func = simulate_func
        self._out: dict[str, Any] | None = None

    # alias to preserve legacy name
    def EMfit(self, **kwargs) -> dict[str, Any]:
        return self.fit(**kwargs).__dict__

    def fit(
        self,
        *,
        verbose: int = 1,
        mstep_maxit: int = 200,
        estep_maxit: int | None = None,
        convergence_method: str = "sum",
        convergence_type: str = "NPL",
        convergence_custom: str | None = None,
        convergence_crit: float = 1e-3,
        convergence_precision: int = 6,
        njobs: int = -1,
        optim_method: str = "BFGS",
        optim_options: dict | None = None,
        max_restarts: int = 2,
        seed: int | None = None,
        prior_mu: np.ndarray | None = None,
        prior_sigma: np.ndarray | None = None,
    ) -> FitResult:
        if self.all_data is None:
            raise ValueError("all_data must be provided to fit the model.")
        config = EMConfig(
            mstep_maxit=mstep_maxit,
            estep_maxit=estep_maxit,
            convergence_method=convergence_method,  # kept for compatibility
            convergence_type="NPL",                 # LME path omitted in this high-level wrapper
            convergence_custom=convergence_custom,
            convergence_crit=convergence_crit,
            convergence_precision=convergence_precision,
            njobs=njobs,
            seed=seed,
        )
        # configure optimizer
        from .core.optim import OptimConfig
        config.optim = OptimConfig(method=optim_method, options=optim_options, max_restarts=max_restarts)

        # optional prior override
        prior = None
        if prior_mu is not None or prior_sigma is not None:
            from .core.priors import GaussianPrior
            if prior_mu is None or prior_sigma is None:
                raise ValueError("Provide both prior_mu and prior_sigma to override the prior.")
            prior = GaussianPrior(mu=np.asarray(prior_mu).reshape(-1), sigma=np.asarray(prior_sigma).reshape(-1))

        out = EMfit(
            all_data=self.all_data,
            objfunc=self.fit_func,
            param_names=self.param_names,
            verbose=verbose,
            config=config,
            prior=prior,
        )
        self._out = out
        return FitResult(
            m=out["m"],
            inv_h=out["inv_h"],
            posterior_mu=out["posterior"]["mu"],
            posterior_sigma=out["posterior"]["sigma"],
            NPL=out["NPL"],
            NLPrior=out["NLPrior"],
            NLL=out["NLL"],
            convergence=out["convergence"],
        )

    def simulate(self, *args, **kwargs):
        if self.simulate_func is None:
            raise AttributeError("No simulate_func provided.")
        return self.simulate_func(*args, **kwargs)

    # utilities convenient for users
    def subject_params(self) -> np.ndarray:
        if self._out is None:
            raise RuntimeError("Call fit() first.")
        return self._out["m"].T.copy()

    def posterior(self) -> dict[str, np.ndarray]:
        if self._out is None:
            raise RuntimeError("Call fit() first.")
        return self._out["posterior"]
