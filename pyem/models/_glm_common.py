"""Shared helpers for the GLM/regression family (glm, glm_decay, logit, logit_decay, glm_ar)."""
from __future__ import annotations
import numpy as np


def _calc_bic(nll: float, nparams: int, nobs: int) -> float:
    """BIC = k*log(n) + 2*NLL, the line every ``*_fit`` in this family repeated by hand."""
    return nparams * np.log(nobs) + 2.0 * nll
