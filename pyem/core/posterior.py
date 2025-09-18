from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass
class RecoveryResult:
    corr: np.ndarray
    rmse: np.ndarray
    est_params: np.ndarray

def parameter_recovery(true_params: np.ndarray, est_params: np.ndarray) -> RecoveryResult:
    """
    Compare true vs. estimated subject-level parameters: return per-parameter Pearson r and RMSE.
    """
    assert true_params.shape == est_params.shape
    dif = est_params - true_params
    rmse = np.sqrt(np.mean(dif**2, axis=0))
    corr = np.array([np.corrcoef(true_params[:, j], est_params[:, j])[0,1] for j in range(true_params.shape[1])])
    return RecoveryResult(corr=corr, rmse=rmse, est_params=est_params)

@dataclass
class IdentifiabilityResult:
    confusion: np.ndarray  # (n_models, n_models)
    winners: np.ndarray    # (n_datasets,)

def model_identifiability(models, param_sets, nblocks: int, ntrials: int, chooser: Callable[[dict], float]):
    """
    Simulate from each model with its param set; fit all models to each simulated dataset and pick winner.
    Returns confusion matrix where rows are generating model and cols are winning model.
    - chooser: function(out_dict) -> scalar score (lower is better)
    """
    n_models = len(models)
    confusion = np.zeros((n_models, n_models), dtype=int)
    winners_all = []
    # generate one dataset per model (can be extended to many)
    for g_idx, (gen_model, gen_params) in enumerate(param_sets):
        sim = gen_model.simulate(gen_params, nblocks=nblocks, ntrials=ntrials)
        all_data = [[c, r] for c, r in zip(sim["choices"], sim["rewards"])]
        scores = []
        for m_idx, mdl in enumerate(models):
            mdl.all_data = all_data  # ensure model fits the current dataset
            res = mdl.fit(mstep_maxit=10, verbose=0, njobs=1)  # small fit for ID check
            scores.append(chooser(res.__dict__))
        win = int(np.argmin(np.asarray(scores)))
        winners_all.append(win)
        confusion[g_idx, win] += 1
    return IdentifiabilityResult(confusion=confusion, winners=np.asarray(winners_all))
