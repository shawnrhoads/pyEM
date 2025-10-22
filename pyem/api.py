
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Sequence
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from .core.em import EMfit, EMConfig
from .core.optim import OptimConfig
from .core.posterior import parameter_recovery
from .core.priors import GaussianPrior, Prior
from .utils import plotting
from .utils.stats import calc_BICint, calc_LME

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
      model = EMModel(all_data, fit_func, param_names, param_xform=[norm2beta, norm2alpha], simulate_func=...)
      result = model.fit(...)
      sim = model.simulate(...)
      
      # Access parameter transformations:
      beta_transform = model.get_param_transform("beta")  # or model.param_xform[0]
      transformed_value = beta_transform(0.5)
    """
    
    def __init__(
        self,
        all_data: Sequence[Sequence[Any]] | None,
        fit_func: Callable[..., float],
        param_names: Sequence[str],
        param_xform: Sequence[Callable] | None = None,
        simulate_func: Callable[..., Any] | None = None,
    ) -> None:
        self.all_data = all_data
        self.fit_func = fit_func
        self.param_names = list(param_names)
        self.simulate_func = simulate_func
        self._out: dict[str, Any] | None = None
        
        # Store parameter transformation functions
        if param_xform is not None:
            if len(param_xform) != len(param_names):
                raise ValueError(f"param_xform length ({len(param_xform)}) must match param_names length ({len(param_names)})")
            self.param_xform = list(param_xform)
        else:
            self.param_xform = None
    
    def get_param_transform(self, param_name_or_index):
        """
        Get parameter transformation function by parameter name or index.
        
        Args:
            param_name_or_index: Parameter name (str) or index (int)
            
        Returns:
            Transformation function for the specified parameter
            
        Raises:
            ValueError: If param_xform was not provided or parameter not found
        """
        if self.param_xform is None:
            raise ValueError("param_xform was not provided to the model")
        
        if isinstance(param_name_or_index, str):
            try:
                index = self.param_names.index(param_name_or_index)
            except ValueError:
                raise ValueError(f"Parameter '{param_name_or_index}' not found in param_names")
        else:
            index = param_name_or_index
            if not (0 <= index < len(self.param_xform)):
                raise ValueError(f"Index {index} out of range for param_xform")
        
        return self.param_xform[index]

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
        njobs: int = -2,
        optim_method: str = "BFGS",
        optim_options: dict | None = None,
        max_restarts: int = 2,
        seed: int | None = None,
        prior_mu: np.ndarray | None = None,
        prior_sigma: np.ndarray | None = None,
        prior: Prior | None = None,
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
        config.optim = OptimConfig(method=optim_method, options=optim_options, max_restarts=max_restarts)

        # optional prior override
        if prior is not None and (prior_mu is not None or prior_sigma is not None):
            raise ValueError("Specify either prior or prior_mu/prior_sigma, not both.")
        if prior_mu is not None or prior_sigma is not None:
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
        # compute final arrays and store for convenience
        self.outfit = self.get_outfit()
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
        params = self._out["m"].T.copy()
        if self.param_xform is not None:
            for i, func in enumerate(self.param_xform):
                params[:, i] = func(params[:, i])
        return params

    def posterior(self) -> dict[str, np.ndarray]:
        if self._out is None:
            raise RuntimeError("Call fit() first.")
        return self._out["posterior"]

    def compute_integrated_bic(self, nsamples: int = 500, func_output: str = "all", nll_key: str = "nll") -> float:
        """
        Compute integrated Bayesian Information Criterion (BICint).
        
        Args:
            nsamples: Number of samples for Monte Carlo integration
            func_output: Output type to request from fit function
            nll_key: Key to extract negative log-likelihood from fit function output
            
        Returns:
            Integrated BIC value
        """
        if self._out is None:
            raise RuntimeError("Call fit() first.")
        
        posterior = self._out["posterior"]
        return calc_BICint(
            self.all_data, 
            self.param_names, 
            posterior["mu"], 
            posterior["sigma"], 
            self.fit_func,
            nsamples=nsamples,
            func_output=func_output,
            nll_key=nll_key
        )

    def compute_lme(self) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Compute Laplace approximation for log model evidence (LME).
        
        Returns:
            Tuple of (Laplace approximation per subject, total LME, good Hessian flags)
        """
        if self._out is None:
            raise RuntimeError("Call fit() first.")
        
        return calc_LME(self._out["inv_h"], self._out["NPL"])

    def get_outfit(self) -> dict[str, np.ndarray]:
        """Return a dictionary of outputs from the ``fit_func`` for each subject."""
        if self._out is None:
            raise RuntimeError("Call fit() first.")

        nsubjects = self._out["m"].shape[1]

        # determine available keys and shapes from first subject
        first_params = self._out["m"][:, 0]
        first_args = self.all_data[0]
        if not isinstance(first_args, (list, tuple)):
            first_args = (first_args,)
        first_fit = self.fit_func(first_params, *first_args, prior=None, output="all")

        arrays_dict: dict[str, np.ndarray] = {}
        for key, value in first_fit.items():
            if isinstance(value, (int, float, np.number)):
                arrays_dict[key] = np.zeros(nsubjects)
            elif isinstance(value, (list, np.ndarray)):
                arr = np.asarray(value)
                if arr.ndim == 0:
                    arrays_dict[key] = np.zeros(nsubjects)
                else:
                    arrays_dict[key] = np.zeros((nsubjects, *arr.shape), dtype=arr.dtype)
            else:
                arrays_dict[key] = np.empty(nsubjects, dtype=object)

        for subj_idx in range(nsubjects):
            params = self._out["m"][:, subj_idx]
            args = self.all_data[subj_idx]
            if not isinstance(args, (list, tuple)):
                args = (args,)
            subj_fit = self.fit_func(params, *args, prior=None, output="all")
            for key in arrays_dict.keys():
                if key in subj_fit:
                    arrays_dict[key][subj_idx] = subj_fit[key]

        return arrays_dict

    def scipy_minimize(self) -> dict[str, Any]:
        """Fit each subject independently using :func:`scipy.optimize.minimize`."""

        if self.all_data is None:
            raise ValueError("all_data must be provided to fit the model.")

        nsubjects = len(self.all_data)
        nparams = len(self.param_names)

        m = np.zeros((nparams, nsubjects))
        inv_h = np.zeros((nparams, nparams, nsubjects))
        NPL = np.zeros(nsubjects)

        for subj_idx, args in enumerate(self.all_data):
            if not isinstance(args, (list, tuple)):
                args = (args,)

            def obj_func(params: np.ndarray) -> float:
                return self.fit_func(params, *args, output="npl")

            x0 = np.random.randn(nparams)
            res = minimize(obj_func, x0=x0, method="BFGS")
            m[:, subj_idx] = res.x
            NPL[subj_idx] = res.fun

            if hasattr(res, "hess_inv"):
                try:
                    inv_h[:, :, subj_idx] = np.asarray(res.hess_inv)
                except Exception:
                    inv_h[:, :, subj_idx] = np.eye(nparams) * max(1.0, np.linalg.norm(res.x) + 1e-6)
            else:
                inv_h[:, :, subj_idx] = np.eye(nparams) * max(1.0, np.linalg.norm(res.x) + 1e-6)

        posterior_sigma = np.array([np.diag(inv_h[:, :, i]) for i in range(nsubjects)]).T

        return {
            "m": m,
            "inv_h": inv_h,
            "posterior": {"mu": m, "sigma": posterior_sigma},
            "NPL": NPL,
            "convergence": True,
            "individual_fit": True,
        }


    def recover(self, true_params: np.ndarray, pr_inputs: List[str], simulate_func: Callable = None,**sim_kwargs) -> dict[str, Any]:
        """
        Parameter recovery analysis given true parameters and simulation function.
        
        Args:
            true_params: True parameter values (nsubjects x nparams)
            simulate_func: Simulation function (uses self.simulate_func if None)
            **sim_kwargs: Additional arguments for simulation
            
        Returns:
            Dictionary containing true params, estimated params, and recovery metrics.
            The ``correlation`` entry provides a Pearson correlation coefficient for
            each parameter (array of length ``nparams``).
        """
        if simulate_func is None:
            simulate_func = self.simulate_func
        if simulate_func is None:
            raise AttributeError("No simulate_func provided.")
        
        # Simulate data with true parameters
        sim = simulate_func(true_params, **sim_kwargs)
        
        # Prepare data for fitting
        missing = [k for k in pr_inputs if k not in sim]
        if missing:
            raise ValueError(
                f"Simulation output is missing expected keys: {missing}. "
                f"Available keys: {list(sim.keys())}"
            )

        # Validate equal lengths for all requested inputs
        lengths = {k: len(sim[k]) for k in pr_inputs}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Inconsistent lengths among requested inputs: {lengths}. "
                "All requested series must be the same length."
            )

        # Build all_data as rows of the selected inputs (e.g., [choice, reward] per trial)
        all_data = [list(row) for row in zip(*(sim[k] for k in pr_inputs))]

        # Create new model instance with simulated data
        recovery_model = EMModel(
            all_data=all_data,
            fit_func=self.fit_func,
            param_names=self.param_names,
            simulate_func=simulate_func,
            param_xform=self.param_xform,
        )
        
        # Fit the model
        fit_result = recovery_model.fit(verbose=0)

        # grab estimated params with transformations applied if applicable
        estimated_params = recovery_model.subject_params()

        corr = parameter_recovery(true_params, estimated_params).corr
        recovery_dict = {
            'true_params': true_params,
            'estimated_params': estimated_params,
            'sim': sim,
            'fit_result': fit_result,
            'correlation': corr,
        }

        self._out = recovery_model._out
        return recovery_dict

    def plot_recovery(
        self,
        recovery_dict: dict,
        show_line: bool = True,
        figsize: tuple | None = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot parameter recovery as scatter plots of simulated vs estimated parameters.
        Creates 3 columns with as many rows as needed, with compact spacing and
        subplot sizes that scale with the grid.
        
        Args:
            recovery_dict: Output from recover() method, containing:
                - 'true_params' (array-like, shape [n_sims, n_params])
                - 'estimated_params' (array-like, shape [n_sims, n_params])
            show_line: Whether to draw x=y line
            figsize: Figure size
            show: Call :func:`matplotlib.pyplot.show` after drawing

        Returns:
            matplotlib Figure object
        """
        true_params = recovery_dict['true_params']
        estimated_params = recovery_dict['estimated_params']
        nparams = true_params.shape[1]

        # Grid: 3 columns, compute rows
        ncols = 3
        nrows = int(np.ceil(nparams / ncols))

        # Figure size: scale per-subplot to avoid tiny axes.
        # Aim for 5x5 inches per subplot (square-ish data area works well here).
        per_ax_w, per_ax_h = 3.5, 3.5
        fig_w = per_ax_w * ncols
        fig_h = per_ax_h * nrows
        if figsize is None:
            figsize = (fig_w, fig_h)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=figsize,
            constrained_layout=True, # let Matplotlib handle spacing
            squeeze=False
        )

        # Fine-tune constrained_layout paddings (reduces big gutters)
        # w_pad/h_pad: padding around the figure edges; wspace/hspace: padding between subplots
        fig.get_layout_engine().set() #h_pad=X, w_pad=Y, hspace=Z, wspace=W

        axes = axes.ravel()
        names = list(self.param_names)[:nparams]

        for i, param_name in enumerate(names):
            ax = axes[i]
            plotting.plot_scatter(
                true_params[:, i], f'True {param_name}',
                estimated_params[:, i], f'Estimated {param_name}',
                ax=ax,
                show_line=show_line,
                equal_limits=True,     # still equalize limits (handled w/ box aspect below)
                s=100,                  # slightly smaller markers to reduce overlap
                alpha=0.6,
                colorname='royalblue',
                annotate=True,
            )
            # Title & tick/label sizing tuned so they don't collide with data
            ax.tick_params(labelsize=12)
            ax.xaxis.label.set_size(12)
            ax.yaxis.label.set_size(12)

            # Keep plots square without blowing up gutters
            # (avoid ax.set_aspect('equal', adjustable='box') here)
            try:
                ax.set_box_aspect(1)   # Matplotlib >=3.4
            except Exception:
                pass

        # Remove unused axes completely so they don't consume layout space
        for j in range(nparams, len(axes)):
            axes[j].remove()

        if show:
            plt.show()

        return fig
