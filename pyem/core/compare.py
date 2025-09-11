
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Sequence, Callable, TYPE_CHECKING
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.stats import calc_LME, calc_BICint, pseudo_r2_from_nll
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
    bicint_kwargs: dict | None = {"nsamples":2000, "func_output":"all", "nll_key":"nll"},
    r2_kwargs: dict | None = None,
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
    
    def identify(
        self,
        rounds: int = 10,
        *,
        sim_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
        bicint_kwargs: dict | None = None,
        r2_kwargs: dict | None = None,
        seed: int | None = None,
        verbose: int = 0,      # NEW: verbosity (0=silent, 1=round/model, 2=per-fit)
    ) -> pd.DataFrame:
        """
        For each round (outer loop), for each EMModel in self.models:
          - simulate using that model's simulate_func
          - fit every model to the simulated data using its fit_func

        Returns a DataFrame with columns:
            ['Simulated','Estimated','LME','BICint','pseudoR2','bestlme','bestbic','bestR2']

        Raises:
            AttributeError if any simulated model has no simulate_func.
        """
        rng = np.random.default_rng(seed)
        sim_kwargs = sim_kwargs or {}
        fit_kwargs = fit_kwargs or {}

        if bicint_kwargs is None:
            bicint_kwargs = {"nsamples": 2000, "func_output": "all", "nll_key": "nll"}
        if r2_kwargs is None:
            r2_kwargs = {}

        names = self.model_names

        # accumulators
        metrics = {
            "LME": {(s, e): [] for s in names for e in names},
            "BICint": {(s, e): [] for s in names for e in names},
            "R2": {(s, e): [] for s in names for e in names},
        }
        best_lme = {(s, e): 0 for s in names for e in names}
        best_bic = {(s, e): 0 for s in names for e in names}
        best_r2  = {(s, e): 0 for s in names for e in names}

        # ---------- OUTER LOOP OVER ROUNDS ----------
        for r in range(rounds):
            if verbose >= 1:
                print(f"[identify] round {r+1}/{rounds}")

            # loop over SIMULATED models
            for sm_i, sim_model in enumerate(self.models):
                sim_name = names[sm_i]

                if getattr(sim_model, "simulate_func", None) is None:
                    raise AttributeError(
                        f"Model '{sim_name}' has no simulate_func; cannot run identifiability."
                    )

                # choose parameters to simulate with
                if getattr(sim_model, "_out", None) is not None:
                    true_params = sim_model._out["m"].T
                else:
                    nsubjects = 10
                    nparams = len(getattr(sim_model, "param_names", [])) or 2
                    true_params = rng.normal(size=(nsubjects, nparams))

                # simulate data
                sim = sim_model.simulate_func(true_params, **sim_kwargs)
                if not (isinstance(sim, dict) and "choices" in sim and "rewards" in sim):
                    raise ValueError(
                        f"Simulation from '{sim_name}' did not return expected keys 'choices' and 'rewards'."
                    )
                all_data = [[c, rw] for c, rw in zip(sim["choices"], sim["rewards"])]

                # fit EVERY model to this simulated dataset
                per_round_LME = []
                per_round_BIC = []
                per_round_R2  = []

                for em_j, est_model in enumerate(self.models):
                    est_name = names[em_j]
                    if verbose >= 2:
                        print(f"  - fitting Estimated='{est_name}' to Simulated='{sim_name}'")

                    # build a fresh EMModel for the estimator
                    tmp = EMModel(
                        all_data=all_data,
                        fit_func=est_model.fit_func,
                        param_names=est_model.param_names,
                        simulate_func=getattr(est_model, "simulate_func", None),
                    )
                    fit_res = tmp.fit(verbose=0, **fit_kwargs)

                    # LME (higher better)
                    lme = None
                    if fit_res.inv_h is not None and fit_res.NPL is not None:
                        _, lme_total, _ = calc_LME(fit_res.inv_h, fit_res.NPL)  # from utils.stats
                        lme = float(lme_total)

                    # BICint (lower better)
                    try:
                        bic = calc_BICint(
                            all_data,
                            est_model.param_names,
                            fit_res.posterior_mu,
                            fit_res.posterior_sigma,
                            est_model.fit_func,
                            **bicint_kwargs,
                        )
                    except Exception:
                        bic = np.nan

                    # pseudo-R² (higher better)
                    try:
                        r2 = pseudo_r2_from_nll(fit_res.NLL, **r2_kwargs)
                    except Exception:
                        r2 = np.nan

                    metrics["LME"][(sim_name, est_name)].append(lme)
                    metrics["BICint"][(sim_name, est_name)].append(bic)
                    metrics["R2"][(sim_name, est_name)].append(r2)

                    per_round_LME.append((est_name, lme))
                    per_round_BIC.append((est_name, bic))
                    per_round_R2.append((est_name, r2))

                # determine winners for THIS (sim_name, round)
                def _winner(pairs, prefer="max"):
                    vals = [(n, v) for (n, v) in pairs if v is not None and not np.isnan(v)]
                    if not vals:
                        return None
                    return (max if prefer == "max" else min)(vals, key=lambda t: t[1])[0]

                w_lme = _winner(per_round_LME, "max")
                w_bic = _winner(per_round_BIC, "min")
                w_r2  = _winner(per_round_R2,  "max")

                if w_lme is not None:
                    best_lme[(sim_name, w_lme)] += 1
                if w_bic is not None:
                    best_bic[(sim_name, w_bic)] += 1
                if w_r2 is not None:
                    best_r2[(sim_name, w_r2)] += 1

        # assemble tidy dataframe
        rows = []
        for sim_name in names:
            for est_name in names:
                LME_mean = (np.nanmean(metrics["LME"][(sim_name, est_name)])
                            if metrics["LME"][(sim_name, est_name)] else np.nan)
                BIC_mean = (np.nanmean(metrics["BICint"][(sim_name, est_name)])
                            if metrics["BICint"][(sim_name, est_name)] else np.nan)
                R2_mean  = (np.nanmean(metrics["R2"][(sim_name, est_name)])
                            if metrics["R2"][(sim_name, est_name)] else np.nan)
                rows.append({
                    "Simulated": sim_name,
                    "Estimated": est_name,
                    "LME": LME_mean,
                    "BICint": BIC_mean,
                    "pseudoR2": R2_mean,
                    "bestlme": best_lme[(sim_name, est_name)],
                    "bestbic": best_bic[(sim_name, est_name)],
                    "bestR2":  best_r2[(sim_name, est_name)],
                })

        df = pd.DataFrame(rows)

        self.identifiability_matrix = df
        self._identify_rounds = rounds
        if verbose >= 1:
            print("[identify] done")
        return df

    def identifiability_analysis(self, *args, **kwargs):
        """Backward-compatible alias for :meth:`identify`."""
        return self.identify(*args, **kwargs)

    def plot_identifiability(
        self,
        df: pd.DataFrame | None = None,
        metric: str = "LME",
        *,
        rounds: int | None = None,
        cmap: str = "viridis",
        annotate: bool = True,
        figsize: tuple = (8, 6),
    ) -> plt.Figure:
        """
        Plot a heatmap of identifiability as the PROPORTION of rounds that the Estimated
        model won for each (Simulated, Estimated) pair, using the winner-count fields
        produced by identify().

        Args:
            df: DataFrame returned by identify(); if None, uses self.identifiability_matrix
            metric: which metric's winners to visualize: {"LME","BICint","pseudoR2"}
            rounds: total rounds per Simulated model (if None, inferred per row by sum of winners)
            cmap: colormap
            annotate: whether to add cell annotations
            figsize: figure size

        Returns:
            matplotlib Figure
        """
        if df is None:
            if isinstance(self.identifiability_matrix, pd.DataFrame):
                df = self.identifiability_matrix
            else:
                raise RuntimeError("Run identify() first or pass a dataframe.")

        # Map metric -> winner-count column produced by identify()
        metric = metric.strip()
        metric_key = metric.upper()
        best_map = {
            "LME": "bestlme",
            "BICINT": "bestbic",
            "PSEUDOR2": "bestR2",
        }
        if metric_key not in best_map:
            raise ValueError("metric must be one of {'LME','BICint','pseudoR2'}")
        best_col = best_map[metric_key]

        # Pivot winner counts to matrix (rows=Simulated, cols=Estimated)
        counts = df.pivot(index="Simulated", columns="Estimated", values=best_col).astype(float)

        # Determine denominators (rounds). Prefer explicit arg, else internal, else infer per-row.
        if rounds is None:
            rounds = getattr(self, "_identify_rounds", None)

        if rounds is not None:
            # Single scalar denominator for all rows
            denom = pd.Series(rounds, index=counts.index, dtype=float)
        else:
            # Infer per Simulated model: sum of winner counts across columns
            # (handles cases with occasional no-winner rounds by yielding < 1.0 max)
            denom = counts.sum(axis=1)
            # avoid divide-by-zero
            denom[denom == 0] = np.nan

        # Compute proportions row-wise
        prop = counts.div(denom, axis=0)

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(prop.values, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)

        # Colorbar labeled as a proportion
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Proportion of rounds won")

        # Axis ticks/labels
        ax.set_xticks(range(prop.shape[1]))
        ax.set_yticks(range(prop.shape[0]))
        ax.set_xticklabels(prop.columns, rotation=45, ha="right")
        ax.set_yticklabels(prop.index)

        # Title and axis labels
        title = {
            "LME": "Log Model Evidence (proportion wins)",
            "BICINT": "Integrated BIC (proportion wins, lower is better)",
            "PSEUDOR2": "Pseudo R² (proportion wins)",
        }[metric_key]
        ax.set_title(title)
        ax.set_xlabel("Estimated")
        ax.set_ylabel("Simulated")

        # Optional per-cell annotations
        if annotate:
            for i in range(prop.shape[0]):
                for j in range(prop.shape[1]):
                    val = prop.iat[i, j]
                    if np.isnan(val):
                        txt = "–"
                    else:
                        txt = f"{val:.2f}"
                    # legible text color
                    ax.text(j, i, txt,
                            ha="center", va="center",
                            color="white" if (not np.isnan(val) and val >= 0.5) else "black")

        plt.tight_layout()
        return fig
