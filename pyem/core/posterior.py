
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Sequence, Any

@dataclass
class PPCResult:
    sims_stats: np.ndarray  # (n_sims,)
    obs_stat: float
    p_value: float

def posterior_predictive_check(
    simulate_func: Callable[..., dict],
    fit_result: dict,
    all_data: Sequence[Sequence[Any]],
    stat_fn: Callable[[Sequence[Sequence[Any]]], float],
    n_sims: int = 200,
    rng: np.random.Generator | None = None,
    assemble_data: Callable[[dict], list] | None = None
) -> PPCResult:
    """
    Simulate datasets from the fitted posterior (Normal approx per subject) and compare a statistic.
    - simulate_func: function(params, ...) -> dict with 'choices'/'rewards' or domain-specific
    - fit_result: dict returned by EMfit
    - stat_fn: maps a dataset (same structure as all_data) -> scalar
    - assemble_data: optional adapter to turn simulate_func output dict into all_data-like list
    """
    if rng is None:
        rng = np.random.default_rng()
    m = fit_result["m"]          # (nparams, nsubjects)
    inv_h = fit_result["inv_h"]  # (nparams, nparams, nsubjects)
    nparams, nsubjects = m.shape
    sims = []
    for _ in range(n_sims):
        samples = np.zeros_like(m.T)  # (nsubjects, nparams)
        for s in range(nsubjects):
            cov = inv_h[:, :, s]
            # numerical guard: ensure PSD
            cov = cov + 1e-8 * np.eye(cov.shape[0])
            samples[s] = rng.multivariate_normal(mean=m[:, s], cov=cov)
        sim_out = simulate_func(samples, **fit_result.get("sim_kwargs", {}))
        sim_data = assemble_data(sim_out) if assemble_data is not None else sim_out
        sims.append(stat_fn(sim_data))
    sims = np.asarray(sims)
    obs = stat_fn(all_data)
    p = float((np.sum(sims >= obs) + 1) / (len(sims) + 1))
    return PPCResult(sims_stats=sims, obs_stat=obs, p_value=p)

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
            res = mdl.fit(mstep_maxit=10, verbose=0, njobs=1)  # small fit for ID check
            scores.append(chooser(res.__dict__))
        win = int(np.argmin(np.asarray(scores)))
        winners_all.append(win)
        confusion[g_idx, win] += 1
    return IdentifiabilityResult(confusion=confusion, winners=np.asarray(winners_all))
