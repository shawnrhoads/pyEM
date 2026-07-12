"""Equal-variance Gaussian signal detection theory (SDT) model.

Implements the classic yes/no old/new recognition-memory task: internal
evidence for studied ("old") items is drawn from ``N(+d'/2, 1)`` and for
novel ("new") items from ``N(-d'/2, 1)``; the subject responds "old" iff
evidence exceeds a single response criterion ``c``. Free parameters are
sensitivity (``dprime``, d' >= 0) and response bias (``criterion``).

Optional extension (NOT implemented in this module): a type-2/confidence
SDT variant that replaces the single criterion with multiple ordered
criteria to jointly model old/new responses and graded confidence ratings.
The required model here stays the plain yes/no dprime/criterion form.
"""
import numpy as np
from scipy.stats import norm
from ..utils.math import norm2beta, calc_fval
from ..core.modelspec import ModelSpec


def sdt_sim(params: np.ndarray, ntrials: int = 200, seed: int | None = None, **kwargs) -> dict:
    """Simulate an equal-variance Gaussian signal-detection (old/new recognition) task.

    Each trial presents either a previously-studied ("old", ``is_old=1``) or
    novel ("new", ``is_old=0``) item. Signal strength (internal evidence) for
    old items is drawn from ``N(+d'/2, 1)`` and for new items from
    ``N(-d'/2, 1)``. The subject responds "old" (``resp_old=1``) whenever the
    sampled evidence exceeds the response criterion ``c``.

    Parameters are supplied in **natural** space: ``params[:, 0]`` = ``dprime``
    (sensitivity, >= 0) and ``params[:, 1]`` = ``criterion`` (real-valued bias).
    """
    nsubjects = params.shape[0]

    all_dprime = params[:, 0]
    all_criterion = params[:, 1]

    # bounds checks
    if not ((all_dprime >= 1e-5) & (all_dprime <= 20.0)).all():
        raise ValueError("dprime values out of bounds")

    rng = np.random.default_rng(seed)

    is_old = np.zeros((nsubjects, ntrials), dtype=float)
    resp_old = np.zeros((nsubjects, ntrials), dtype=float)
    evidence = np.zeros((nsubjects, ntrials), dtype=float)

    for s in range(nsubjects):
        dprime = float(all_dprime[s])
        criterion = float(all_criterion[s])

        # half old / half new (shuffled) per subject
        trial_is_old = np.zeros(ntrials, dtype=float)
        trial_is_old[: ntrials // 2] = 1.0
        rng.shuffle(trial_is_old)
        # if ntrials is odd, decide the leftover trial at random
        if ntrials % 2 == 1:
            trial_is_old[-1] = float(rng.integers(0, 2))

        mean = np.where(trial_is_old == 1.0, dprime / 2.0, -dprime / 2.0)
        ev = rng.normal(loc=mean, scale=1.0, size=ntrials)
        resp = (ev > criterion).astype(float)

        is_old[s, :] = trial_is_old
        resp_old[s, :] = resp
        evidence[s, :] = ev

    return {
        "params": np.array([all_dprime, all_criterion]).T,
        "is_old": is_old,
        "resp_old": resp_old,
        "evidence": evidence,
    }


def sdt_fit(params, is_old, resp_old, prior=None, output="npl"):
    """
    A thin adapter compatible with EM: returns NPL or NLL.
    params: (2,) in normalized space -> [dprime, criterion]

    P(resp_old=1 | is_old) = Phi(d'/2 - c)   if is_old == 1  (hit rate)
    P(resp_old=1 | is_old) = Phi(-d'/2 - c)  if is_old == 0  (false-alarm rate)
    """
    dprime = float(norm2beta(params[0]))
    criterion = float(params[1])  # identity transform (real-valued)

    # reject values outside natural bounds
    if not (1e-5 <= dprime <= 20.0):
        return 1e7

    is_old = np.asarray(is_old, dtype=float)
    resp_old = np.asarray(resp_old, dtype=float)

    signal = np.where(is_old == 1.0, dprime / 2.0 - criterion, -dprime / 2.0 - criterion)
    p_old = norm.cdf(signal)
    p_old = np.clip(p_old, 1e-12, 1 - 1e-12)

    nll = -np.sum(resp_old * np.log(p_old) + (1.0 - resp_old) * np.log(1.0 - p_old))

    if output == "all":
        return {
            "params": [dprime, criterion],
            "is_old": is_old,
            "resp_old": resp_old,
            "p_old": p_old,
            "nll": nll,
        }

    # otherwise compute objective value
    return calc_fval(nll, params, prior=prior, output=output)


sdt_desc = """Equal-variance Gaussian signal detection theory (SDT) model of an
old/new recognition memory task. Internal evidence for studied ("old") items
is drawn from N(+d'/2, 1) and for novel ("new") items from N(-d'/2, 1); the
subject responds "old" whenever evidence exceeds a response criterion c.
Free parameters: sensitivity (dprime, d' >= 0), response bias (criterion, c).

Optional extension (not implemented here): a type-2/confidence variant that
replaces the single yes/no criterion with multiple ordered criteria to model
graded confidence ratings alongside old/new responses."""
sdt_id = "sdt"
sdt_spec = {"sdt": {"gaussian_equal_variance": ["dprime", "criterion"]}}
sdt_model = ModelSpec(
    id=sdt_id, spec=sdt_spec, desc=sdt_desc.strip(),
    params=None, sim=sdt_sim, fit=sdt_fit,
)
