
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Sequence, Callable, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
from ..utils.stats import calc_LME, calc_BICint, pseudo_r2_from_nll

if TYPE_CHECKING:
    from ..api import EMModel

@dataclass
class ComparisonRow:
    name: str
    LME: float | None
    BICint: float | None
    R2: float | None

def compare_models(
    models,  # list of EMModel (already fit) or tuples (name, FitResult, extras)
    metric_order: Sequence[str] = ("LME", "BICint", "R2"),
    r2_kwargs: dict | None = None,
    bicint_kwargs: dict | None = None,
) -> List[ComparisonRow]:
    rows: List[ComparisonRow] = []
    for item in models:
        if hasattr(item, "fit_func") or hasattr(item, "fit"):  # EMModel instance
            name = getattr(item, "name", item.fit_func.__name__ if hasattr(item, "fit_func") else "model")
            out = item._out or {}
            all_data = item.all_data
            fit_func = item.fit_func
        else:
            name, out, all_data, fit_func = item  # explicit tuple
        LME = None
        if "inv_h" in out and "NPL" in out:
            _, lme, _ = calc_LME(out["inv_h"], out["NPL"])
            LME = float(lme)
        BICint = None
        if out.get("posterior") is not None and bicint_kwargs is not None:
            mu = out["posterior"]["mu"]; sigma = out["posterior"]["sigma"]
            BICint = calc_BICint(all_data, out.get("param_names", []), mu, sigma, fit_func, **bicint_kwargs)
        R2 = None
        if r2_kwargs is not None and "NLL" in out:
            R2 = pseudo_r2_from_nll(out["NLL"], **r2_kwargs)
        rows.append(ComparisonRow(name=name, LME=LME, BICint=BICint, R2=R2))
    # sort by the first available metric
    for metric in metric_order:
        vals = [getattr(r, metric) for r in rows]
        if any(v is not None for v in vals):
            rev = True if metric in ("LME", "R2") else False  # higher better for LME,R2; lower better for BICint
            rows.sort(key=lambda r: (float("inf") if getattr(r, metric) is None else getattr(r, metric)), reverse=rev)
            break
    return rows


class ModelComparison:
    """
    Class for performing model comparison and identifiability analysis.
    """
    
    def __init__(self, models: List[Any], model_names: List[str] = None):
        """
        Initialize ModelComparison.
        
        Args:
            models: List of EMModel instances or model tuples
            model_names: Optional list of model names
        """
        self.models = models
        self.model_names = model_names or [f"Model_{i+1}" for i in range(len(models))]
        self.comparison_results = None
        self.identifiability_matrix = None
        
    def compare(self, **kwargs) -> List[ComparisonRow]:
        """
        Run model comparison.
        
        Args:
            **kwargs: Arguments passed to compare_models function
            
        Returns:
            List of comparison results
        """
        self.comparison_results = compare_models(self.models, **kwargs)
        return self.comparison_results
    
    def identifiability_analysis(self, n_simulations: int = 100, **sim_kwargs) -> np.ndarray:
        """
        Perform model identifiability analysis by simulating data from each model
        and seeing how often each model is correctly identified as best fitting.
        
        Args:
            n_simulations: Number of simulations per model
            **sim_kwargs: Arguments for simulation
            
        Returns:
            Confusion matrix of identifiability (n_models x n_models)
        """
        n_models = len(self.models)
        confusion_matrix = np.zeros((n_models, n_models))
        
        for true_model_idx, true_model in enumerate(self.models):
            print(f"Testing identifiability for {self.model_names[true_model_idx]}...")
            
            for sim_idx in range(n_simulations):
                # Get the true model fit result or parameters
                if hasattr(true_model, '_out') and true_model._out is not None:
                    # Use fitted parameters from the true model
                    true_params = true_model._out["m"].T
                elif hasattr(true_model, 'subject_params'):
                    try:
                        true_params = true_model.subject_params()
                    except RuntimeError:
                        # Need to fit first or simulate some parameters
                        nsubjects = 10
                        nparams = len(true_model.param_names) if hasattr(true_model, 'param_names') else 2
                        true_params = np.random.randn(nsubjects, nparams)
                else:
                    # Generate random parameters
                    nsubjects = 10
                    nparams = 2  # default
                    true_params = np.random.randn(nsubjects, nparams)
                
                # Simulate data from true model
                if hasattr(true_model, 'simulate_func') and true_model.simulate_func is not None:
                    sim_data = true_model.simulate_func(true_params, **sim_kwargs)
                    if 'choices' in sim_data and 'rewards' in sim_data:
                        all_data = [[c, r] for c, r in zip(sim_data["choices"], sim_data["rewards"])]
                    else:
                        continue  # Skip if simulation doesn't return expected format
                else:
                    continue  # Skip if no simulation function
                
                # Fit all models to simulated data
                model_metrics = []
                for model_idx, model in enumerate(self.models):
                    try:
                        # Create temporary model with simulated data
                        if hasattr(model, 'fit_func'):
                            from ..api import EMModel
                            temp_model = EMModel(
                                all_data=all_data,
                                fit_func=model.fit_func,
                                param_names=model.param_names,
                                simulate_func=getattr(model, 'simulate_func', None)
                            )
                            fit_result = temp_model.fit(mstep_maxit=5, verbose=0)
                            metric = np.sum(fit_result.NPL)  # Use NPL as metric (lower is better)
                        else:
                            # Handle tuple format
                            from ..api import EMModel
                            name, out, orig_data, fit_func = model
                            temp_model = EMModel(all_data=all_data, fit_func=fit_func, param_names=out.get('param_names', ['p1', 'p2']))
                            fit_result = temp_model.fit(mstep_maxit=5, verbose=0)
                            metric = np.sum(fit_result.NPL)
                        
                        model_metrics.append(metric)
                    except Exception as e:
                        model_metrics.append(np.inf)  # High penalty for failed fits
                
                # Find best fitting model (lowest metric)
                best_model_idx = np.argmin(model_metrics)
                confusion_matrix[true_model_idx, best_model_idx] += 1
        
        # Normalize by number of simulations
        confusion_matrix /= n_simulations
        self.identifiability_matrix = confusion_matrix
        return confusion_matrix
    
    def plot_identifiability(self, figsize: tuple = (8, 6), cmap: str = 'Blues') -> plt.Figure:
        """
        Plot confusion matrix of model identifiability.
        
        Args:
            figsize: Figure size
            cmap: Colormap for the heatmap
            
        Returns:
            matplotlib Figure object
        """
        if self.identifiability_matrix is None:
            raise RuntimeError("Run identifiability_analysis() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(self.identifiability_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Proportion correctly identified')
        
        # Set ticks and labels
        ax.set_xticks(range(len(self.model_names)))
        ax.set_yticks(range(len(self.model_names)))
        ax.set_xticklabels(self.model_names)
        ax.set_yticklabels(self.model_names)
        
        # Add text annotations
        for i in range(len(self.model_names)):
            for j in range(len(self.model_names)):
                text = ax.text(j, i, f'{self.identifiability_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black" if self.identifiability_matrix[i, j] < 0.5 else "white")
        
        ax.set_xlabel('Best fitting model')
        ax.set_ylabel('True model')
        ax.set_title('Model Identifiability Confusion Matrix')
        
        plt.tight_layout()
        return fig
