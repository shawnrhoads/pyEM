
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Sequence, Callable, TYPE_CHECKING
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    bicint_kwargs: dict | None = {"nsamples":2000, "func_output":"all", "nll_key":"nll"},
    r2_kwargs: dict | None = {"ntrials": int, "nopts": int},
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
    ) -> pd.DataFrame:
        """
        For each provided EMModel, simulate behavior (rounds times) using its
        simulate_func, then fit each model to those simulated data using the
        corresponding fit_func. Returns a DataFrame with:

        ['Simulated','Estimated','LME','BICint','pseudoR2','bestlme','bestbic','bestR2']

        Notes:
        - 'bestlme' counts (0..rounds) how many rounds the Estimated model
          had the highest LME on data simulated by Simulated model.
        - 'bestbic' counts (0..rounds) how many rounds the Estimated model
          had the lowest BICint.
        - 'bestR2'  counts (0..rounds) how many rounds the Estimated model
          had the highest pseudoR2.
        """
        rng = np.random.default_rng(seed)
        sim_kwargs = sim_kwargs or {}
        fit_kwargs = fit_kwargs or {}
        # sensible defaults that mirror compare_models() behavior
        if bicint_kwargs is None:
            bicint_kwargs = {"nsamples": 2000, "func_output": "all", "nll_key": "nll"}
        # r2_kwargs should contain what's needed by pseudo_r2_from_nll (e.g., ntrials, nopts)
        # provide an empty dict if caller prefers that pseudo_r2_from_nll infers from context
        if r2_kwargs is None:
            r2_kwargs = {}

        n_models = len(self.models)
        names = self.model_names

        # Accumulators keyed by (sim_name, est_name)
        metrics = {
            "LME": {(s, e): [] for s in names for e in names},
            "BICint": {(s, e): [] for s in names for e in names},
            "R2": {(s, e): [] for s in names for e in names},
        }
        # winner counters
        best_lme = {(s, e): 0 for s in names for e in names}
        best_bic = {(s, e): 0 for s in names for e in names}
        best_r2  = {(s, e): 0 for s in names for e in names}

        from ..api import EMModel  # use your high-level wrapper  :contentReference[oaicite:2]{index=2}

        for sm_i, sim_model in enumerate(self.models):
            sim_name = names[sm_i]
            if getattr(sim_model, "simulate_func", None) is None:
                # skip models without simulate_func
                continue

            for r in range(rounds):
                # ----- simulate parameters -----
                # If the sim_model already has a fit, reuse its per-subject params shape.
                if getattr(sim_model, "_out", None) is not None:
                    true_params = sim_model._out["m"].T
                else:
                    # otherwise, draw a reasonable shape
                    # try to infer from a previous model, else default (nsubjects=10, nparams = len(param_names))
                    nsubjects = 10
                    nparams = len(getattr(sim_model, "param_names", [])) or 2
                    true_params = rng.normal(size=(nsubjects, nparams))

                # ----- simulate data -----
                sim = sim_model.simulate_func(true_params, **sim_kwargs)
                if not (isinstance(sim, dict) and "choices" in sim and "rewards" in sim):
                    # If your simulator returns a different structure, adapt here.
                    continue
                all_data = [[c, rwd] for c, rwd in zip(sim["choices"], sim["rewards"])]

                # ----- fit all models to this simulated dataset -----
                per_round_LME = []
                per_round_BIC = []
                per_round_R2  = []
                for em_j, est_model in enumerate(self.models):
                    est_name = names[em_j]

                    # Build a temporary EMModel for the estimated model with this simulated data
                    tmp = EMModel(
                        all_data=all_data,
                        fit_func=est_model.fit_func,
                        param_names=est_model.param_names,
                        simulate_func=getattr(est_model, "simulate_func", None),
                    )
                    # short, quiet fit unless caller overrides
                    fit_res = tmp.fit(verbose=0, **fit_kwargs)

                    # ---- metrics ----
                    # LME via Laplace (higher is better)
                    lme = None
                    if fit_res.inv_h is not None and fit_res.NPL is not None:
                        _, lme_total, _ = calc_LME(fit_res.inv_h, fit_res.NPL)
                        lme = float(lme_total)

                    # BICint (lower is better)
                    bic = None
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

                    # pseudo R^2 (higher is better) — requires caller-provided kwargs
                    r2 = None
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

                # ----- decide winners for this round -----
                # LME: max; BICint: min; R2: max
                def _winner(pairs, prefer="max"):
                    # pairs: list of (name, value)
                    vals = [(n, v) for (n, v) in pairs if v is not None and not np.isnan(v)]
                    if not vals:
                        return None
                    if prefer == "max":
                        return max(vals, key=lambda t: t[1])[0]
                    else:
                        return min(vals, key=lambda t: t[1])[0]

                w_lme = _winner(per_round_LME, "max")
                w_bic = _winner(per_round_BIC, "min")
                w_r2  = _winner(per_round_R2,  "max")

                if w_lme is not None:
                    best_lme[(sim_name, w_lme)] += 1
                if w_bic is not None:
                    best_bic[(sim_name, w_bic)] += 1
                if w_r2 is not None:
                    best_r2[(sim_name, w_r2)] += 1

        # ----- build output DataFrame -----
        rows = []
        for sim_name in names:
            for est_name in names:
                LME_vals = metrics["LME"][(sim_name, est_name)]
                BIC_vals = metrics["BICint"][(sim_name, est_name)]
                R2_vals  = metrics["R2"][(sim_name, est_name)]

                # aggregate across rounds (mean is a stable summary)
                LME_mean = np.nanmean(LME_vals) if len(LME_vals) else np.nan
                BIC_mean = np.nanmean(BIC_vals) if len(BIC_vals) else np.nan
                R2_mean  = np.nanmean(R2_vals)  if len(R2_vals)  else np.nan

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
        self.identifiability_matrix = df  # keep for plotting compatibility
        return df
    
    def plot_identifiability(
        self,
        df: pd.DataFrame | None = None,
        metrics: list[str] = ("LME", "BICint", "pseudoR2"),
    ):
        """
        Plot heatmaps for the given identifiability dataframe.
        Rows = Simulated (actual), Cols = Estimated (fitted).
        """
        if df is None:
            if isinstance(self.identifiability_matrix, pd.DataFrame):
                df = self.identifiability_matrix
            else:
                raise RuntimeError("Run identify() first or pass a dataframe.")

        matrices = {}
        for metric in metrics:
            # pivot to Simulated x Estimated
            mat = df.pivot(index="Simulated", columns="Estimated", values=metric)
            matrices[metric] = mat

        for metric in metrics:
            with sns.plotting_context(context='paper', font_scale=2, rc={'axes.linewidth': 2}):
                m = matrices[metric].astype(float)
                ax = sns.heatmap(m, annot=False, cmap='viridis')
                plt.xticks(rotation=45, ha='right')
                title = {
                    "BICint": "Integrated BIC",
                    "pseudoR2": "Pseudo R²",
                    "LME": "Log Model Evidence",
                }.get(metric, metric)
                plt.title(title)
                ax.set_xlabel('Estimated', fontsize=10)
                ax.set_ylabel('Simulated', fontsize=10)
                plt.tight_layout()
                plt.show()
