from __future__ import annotations
import numpy as np
from dataclasses import dataclass

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

