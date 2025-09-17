
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize

from .core.em import EMfit, EMConfig
from .core.optim import OptimConfig
from .core.posterior import PCCResult, parameter_recovery
from .core.priors import GaussianPrior
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
        self._pcc_result: PCCResult | None = None
        
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
        prior: Any | None = None,
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

        if prior is not None and not isinstance(prior, GaussianPrior):
            out = self.scipy_minimize(prior=prior)
        else:
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
        first_fit = dict(self.fit_func(first_params, *first_args, prior=None, output="all"))

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
            subj_fit = dict(self.fit_func(params, *args, prior=None, output="all"))
            for key in arrays_dict.keys():
                if key in subj_fit:
                    arrays_dict[key][subj_idx] = subj_fit[key]

        return arrays_dict

    def pcc(
        self,
        output_key: str,
        *,
        n_sims: int = 200,
        sim_kwargs: dict[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ) -> PCCResult:
        """Run a posterior predictive check for a selected model output."""

        if self._out is None:
            raise RuntimeError("Call fit() before running a posterior predictive check.")
        if self.simulate_func is None:
            raise AttributeError("No simulate_func provided; posterior predictive checks are unavailable.")

        sim_kwargs = {} if sim_kwargs is None else dict(sim_kwargs)
        rng = np.random.default_rng() if rng is None else rng

        observed_outputs = self.get_outfit()
        if output_key not in observed_outputs:
            raise KeyError(f"Output '{output_key}' was not produced by the fit function.")

        def _ensure_numeric(name: str, value: Any) -> np.ndarray:
            arr = np.asarray(value)
            try:
                return arr.astype(float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Output '{name}' must be numeric for posterior predictive checks.") from exc

        observed = _ensure_numeric(output_key, observed_outputs[output_key])
        nsubjects = observed.shape[0]

        posterior_mean = np.asarray(self._out["m"], dtype=float)
        posterior_cov = np.asarray(self._out["inv_h"], dtype=float)
        if posterior_mean.shape[1] != nsubjects:
            raise ValueError("Posterior mean dimension does not match number of subjects.")
        nparams = posterior_mean.shape[0]

        simulations: list[np.ndarray] = []
        stats = np.zeros(n_sims, dtype=float)
        posterior_samples = np.zeros((n_sims, nsubjects, nparams), dtype=float)

        for sim_idx in range(n_sims):
            samples = np.zeros((nsubjects, nparams), dtype=float)
            for subj_idx in range(nsubjects):
                cov = posterior_cov[:, :, subj_idx]
                cov = 0.5 * (cov + cov.T)
                cov += 1e-8 * np.eye(nparams)
                try:
                    samples[subj_idx] = rng.multivariate_normal(mean=posterior_mean[:, subj_idx], cov=cov)
                except np.linalg.LinAlgError:
                    jitter = 1e-6 * np.eye(nparams)
                    samples[subj_idx] = rng.multivariate_normal(mean=posterior_mean[:, subj_idx], cov=cov + jitter)

            posterior_samples[sim_idx] = samples
            sim_out = dict(self.simulate_func(samples, **sim_kwargs))
            if output_key not in sim_out:
                raise KeyError(f"Simulation output does not contain '{output_key}'.")

            sim_data = _ensure_numeric(output_key, sim_out[output_key])
            if sim_data.shape[0] != nsubjects:
                raise ValueError("Simulation output must match the number of fitted subjects.")

            simulations.append(sim_data)
            stats[sim_idx] = float(np.nanmean(sim_data))

        simulated = np.asarray(simulations, dtype=float)
        obs_stat = float(np.nanmean(observed))
        p_value = float((np.sum(stats >= obs_stat) + 1) / (n_sims + 1))

        result = PCCResult(
            output_key=output_key,
            observed=observed,
            simulated=simulated,
            stats=stats,
            obs_stat=obs_stat,
            p_value=p_value,
            posterior_samples=posterior_samples,
        )
        self._pcc_result = result
        return result

    def plot_pcc(
        self,
        result: PCCResult | None = None,
        *,
        agent_index: int | None = None,
        plot_all_agents: bool = False,
        figsize: tuple[float, float] = (15, 4),
        show: bool = True,
    ) -> plt.Figure:
        """Plot summary diagnostics for a posterior predictive check."""

        if result is None:
            if self._pcc_result is None:
                raise RuntimeError("Run pcc() first or provide a PCCResult to plot.")
            result = self._pcc_result

        observed = np.asarray(result.observed, dtype=float)
        simulated = np.asarray(result.simulated, dtype=float)

        observed_flat = observed.reshape(observed.shape[0], -1)
        simulated_flat = simulated.reshape(simulated.shape[0], simulated.shape[1], -1)

        trial_index = np.arange(observed_flat.shape[1])
        model_means = simulated_flat.mean(axis=1)

        df_model = pd.DataFrame(
            {
                "Trial": np.tile(trial_index, model_means.shape[0]),
                "Value": model_means.reshape(-1),
                "Source": "Model",
            }
        )
        df_data = pd.DataFrame(
            {
                "Trial": np.tile(trial_index, observed_flat.shape[0]),
                "Value": observed_flat.reshape(-1),
                "Source": "Data",
            }
        )
        df_plot = pd.concat([df_model, df_data], ignore_index=True)

        show_agent_column = plot_all_agents or agent_index is not None
        ncols = 3 if show_agent_column else 2
        fig, axes = plt.subplots(1, ncols, figsize=figsize, squeeze=False)
        axes = axes.ravel()

        sns.lineplot(
            data=df_plot,
            x="Trial",
            y="Value",
            hue="Source",
            estimator="mean",
            errorbar="sd",
            ax=axes[0],
        )
        axes[0].set_title(f"Trial means: {result.output_key}")
        axes[0].set_ylabel(result.output_key)
        axes[0].set_xlabel("Trial")
        axes[0].legend(loc="best")

        grand_model = simulated_flat.mean(axis=(1, 2))
        sns.histplot(grand_model, ax=axes[1], color="tab:purple", edgecolor="white")
        axes[1].axvline(result.obs_stat, color="black", linestyle="--", label="Observed mean")
        axes[1].set_title("Grand average distribution")
        axes[1].set_xlabel(result.output_key)
        axes[1].set_ylabel("Frequency")
        axes[1].legend(loc="best")

        if show_agent_column:
            if plot_all_agents:
                indices = np.arange(observed_flat.shape[0])
            else:
                if agent_index is None:
                    raise ValueError("Provide agent_index when plot_all_agents is False.")
                if not 0 <= agent_index < observed_flat.shape[0]:
                    raise IndexError("agent_index is out of range.")
                indices = np.asarray([agent_index])

            subject_predictions = simulated_flat.mean(axis=0)
            df_subject = pd.DataFrame(
                {
                    "Observed": observed_flat[indices].reshape(-1),
                    "Model": subject_predictions[indices].reshape(-1),
                    "Subject": np.repeat(indices, observed_flat.shape[1]),
                }
            )

            scatter_ax = axes[2]
            if len(indices) == 1:
                sns.scatterplot(data=df_subject, x="Observed", y="Model", ax=scatter_ax)
            else:
                df_subject["Subject"] = df_subject["Subject"].astype(str)
                sns.scatterplot(
                    data=df_subject,
                    x="Observed",
                    y="Model",
                    hue="Subject",
                    ax=scatter_ax,
                )

            min_val = float(min(df_subject["Observed"].min(), df_subject["Model"].min()))
            max_val = float(max(df_subject["Observed"].max(), df_subject["Model"].max()))
            scatter_ax.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--")
            scatter_ax.set_title("Agent predictions")
            scatter_ax.set_xlabel("Observed")
            scatter_ax.set_ylabel("Model")

            if len(indices) == 1 and scatter_ax.get_legend() is not None:
                scatter_ax.get_legend().remove()

        if show:
            plt.show()

        return fig

    def scipy_minimize(self, prior: Any | None = None) -> dict[str, Any]:
        """Fit each subject independently using :func:`scipy.optimize.minimize`.

        Args:
            prior: Prior distribution applied to each subject independently.
        """
        if self.all_data is None:
            raise ValueError("all_data must be provided to fit the model.")

        nsubjects = len(self.all_data)
        nparams = len(self.param_names)

        m = np.zeros((nparams, nsubjects))
        inv_h = np.zeros((nparams, nparams, nsubjects))
        NPL = np.zeros(nsubjects)
        NLPrior = np.zeros(nsubjects)

        for subj_idx, args in enumerate(self.all_data):
            if not isinstance(args, (list, tuple)):
                args = (args,)

            def obj_func(params: np.ndarray) -> float:
                return self.fit_func(params, *args, prior=prior, output="npl")

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
            if prior is not None:
                NLPrior[subj_idx] = -prior.logpdf(res.x)

        NLL = NPL - NLPrior
        posterior_sigma = np.array([np.diag(inv_h[:, :, i]) for i in range(nsubjects)]).T

        return {
            "m": m,
            "inv_h": inv_h,
            "posterior": {"mu": m, "sigma": posterior_sigma},
            "NPL": NPL,
            "NLPrior": NLPrior,
            "NLL": NLL,
            "convergence": True,
            "individual_fit": True,
        }

    def recover(self, true_params: np.ndarray, simulate_func: Callable = None, **sim_kwargs) -> dict[str, Any]:
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
        estimated_params = recovery_model.subject_params()

        true_params = sim['params']
        corr = parameter_recovery(true_params, estimated_params).corr
        recovery_dict = {
            'true_params': true_params,
            'estimated_params': estimated_params,
            'sim': sim,
            'fit_result': fit_result,
            'correlation': corr,
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
