
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Sequence
import numpy as np
import matplotlib.pyplot as plt
from .core.em import EMfit, EMConfig
from .core.priors import GaussianPrior
from .utils.stats import calc_BICint, calc_LME
from .utils import plotting

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

    def calculate_final_arrays(self) -> dict[str, np.ndarray]:
        """
        Calculate final arrays based on estimated parameters.
        Generic implementation that works with any fit_func output.
        
        Returns:
            Dictionary containing calculated arrays for each subject
        """
        if self._out is None:
            raise RuntimeError("Call fit() first.")
        
        nsubjects = self._out["m"].shape[1]
        
        # Get first subject's fit to determine available keys and shapes
        first_subj_params = self._out["m"][:, 0]
        first_args = self.all_data[0]
        if not isinstance(first_args, (list, tuple)):
            first_args = (first_args,)
        
        # Get first subject fit with all outputs to determine structure
        first_subj_fit = self.fit_func(first_subj_params, *first_args, prior=None, output='all')
        
        # Initialize arrays_dict based on what the fit_func actually returns
        arrays_dict = {}
        
        for key, value in first_subj_fit.items():
            if isinstance(value, (int, float, np.number)):
                # Scalar values - create 1D array for subjects
                arrays_dict[key] = np.zeros(nsubjects)
            elif isinstance(value, (list, np.ndarray)):
                # Array values - create arrays with subject dimension
                value_array = np.asarray(value)
                if value_array.ndim == 0:
                    # 0-d array (scalar)
                    arrays_dict[key] = np.zeros(nsubjects)
                else:
                    # Multi-dimensional array - add subject dimension
                    subject_shape = (nsubjects,) + value_array.shape
                    arrays_dict[key] = np.zeros(subject_shape, dtype=value_array.dtype)
            else:
                # For other types (strings, objects), create object array
                arrays_dict[key] = np.empty(nsubjects, dtype=object)
        
        # Fill arrays for all subjects
        for subj_idx in range(nsubjects):
            subj_params = self._out["m"][:, subj_idx]
            args = self.all_data[subj_idx]
            if not isinstance(args, (list, tuple)):
                args = (args,)
            
            # Get subject fit with all outputs
            subj_fit = self.fit_func(subj_params, *args, prior=None, output='all')
            
            # Populate arrays based on what's available in subj_fit
            for key in arrays_dict.keys():
                if key in subj_fit:
                    arrays_dict[key][subj_idx] = subj_fit[key]
        
        return arrays_dict

    def fit_individual_nll(self, use_emfit: bool = True) -> dict[str, Any]:
        """
        Fit using either EMfit() or individual negative log likelihood per subject.
        
        Args:
            use_emfit: If True, use EMfit(); if False, fit each subject individually
            
        Returns:
            Dictionary with fit results
        """
        if use_emfit:
            # Use standard EMfit
            return self.fit().__dict__
        else:
            # Fit each subject individually
            if self.all_data is None:
                raise ValueError("all_data must be provided to fit the model.")
            
            nsubjects = len(self.all_data)
            nparams = len(self.param_names)
            
            m = np.zeros((nparams, nsubjects))
            NPL = np.zeros(nsubjects)
            
            for subj_idx, args in enumerate(self.all_data):
                if not isinstance(args, (list, tuple)):
                    args = (args,)
                
                # Simple optimization for individual subjects
                from scipy.optimize import minimize
                
                def obj_func(params):
                    return self.fit_func(params, *args, prior=None, output='nll')
                
                # Random starting point
                x0 = np.random.randn(nparams)
                res = minimize(obj_func, x0=x0, method='BFGS')
                
                m[:, subj_idx] = res.x
                NPL[subj_idx] = res.fun
            
            return {
                'm': m,
                'NPL': NPL,
                'convergence': True,
                'individual_fit': True
            }

    def recover(self, true_params: np.ndarray, simulate_func: Callable = None, **sim_kwargs) -> dict[str, Any]:
        """
        Parameter recovery analysis given true parameters and simulation function.
        
        Args:
            true_params: True parameter values (nsubjects x nparams)
            simulate_func: Simulation function (uses self.simulate_func if None)
            **sim_kwargs: Additional arguments for simulation
            
        Returns:
            Dictionary containing true params, estimated params, and recovery metrics
        """
        if simulate_func is None:
            simulate_func = self.simulate_func
        if simulate_func is None:
            raise AttributeError("No simulate_func provided.")
        
        # Simulate data with true parameters
        sim = simulate_func(true_params, **sim_kwargs)
        
        # Prepare data for fitting
        if 'choices' in sim and 'rewards' in sim:
            all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
        else:
            raise ValueError("Simulation must return 'choices' and 'rewards' keys")
        
        # Create new model instance with simulated data
        recovery_model = EMModel(
            all_data=all_data,
            fit_func=self.fit_func,
            param_names=self.param_names,
            simulate_func=simulate_func
        )
        
        # Fit the model
        fit_result = recovery_model.fit(verbose=0)
        
        # grab estimated params 
        out_fit = recovery_model.calculate_final_arrays()
        estimated_params = out_fit['params']
        
        # Calculate recovery metrics
        dif = estimated_params - true_params
        recovery_dict = {
            'true_params': sim['params'],
            'estimated_params': estimated_params,
            'sim': sim,
            'fit_result': fit_result,
            'correlation': np.corrcoef(true_params.flatten(), estimated_params.flatten())[0, 1],
            'rmse': np.sqrt(np.mean(dif**2, axis=0)),
            'mae': np.mean(np.abs(dif), axis=0),
        }

        return recovery_dict

    def plot_recovery(self, recovery_dict: dict, show_line: bool = True,
                      figsize: tuple = (10, 4), show: bool = True) -> plt.Figure:
        """
        Plot parameter recovery as scatter plots of simulated vs estimated parameters.

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

        # Create 1 x nparams layout (keep squeeze=False to always get 2D array, then ravel)
        fig, axes = plt.subplots(1, nparams, figsize=figsize, squeeze=False)
        axes = axes.ravel()

        # In case self.param_names is longer than nparams
        names = list(self.param_names)[:nparams]

        for i, param_name in enumerate(names):
            ax = axes[i]

            # Use the shared plotting helper
            plotting.plot_scatter(
                true_params[:, i], f'True {param_name}',
                estimated_params[:, i], f'Estimated {param_name}',
                ax=ax,
                show_line=show_line,
                equal_limits=True,
                s=75,
                alpha=0.6,
                colorname='royalblue',
                annotate=True,
            )

            # Title
            ax.set_title(f'{param_name}')

        # Hide any unused axes (just in case)
        for j in range(nparams, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        if show:
            plt.show()
        return fig
