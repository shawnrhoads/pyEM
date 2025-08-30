
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Sequence, Callable
import numpy as np
from ..utils.stats import calc_LME, calc_BICint, pseudo_r2_from_nll

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
