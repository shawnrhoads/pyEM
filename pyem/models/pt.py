"""Prospect Theory model for choices between a certain amount and a gamble.

Implements the value function and one-parameter probability weighting
function from cumulative prospect theory (Tversky & Kahneman, 1992):

    Value function:
        v(x) = x**alpha                    if x >= 0
        v(x) = -lambda * (-x)**beta        if x < 0

    Probability weighting (one-parameter TK1992 form):
        w(p) = p**gamma / (p**gamma + (1 - p)**gamma)**(1/gamma)

Each trial presents a CERTAIN amount against a two-outcome GAMBLE with
outcomes ``o1``/``o2`` occurring with probabilities ``p1``/``p2 = 1 - p1``
(outcomes may be mixed gain/loss). The gamble's subjective value is

    V_g = w(p1) * v(o1) + w(p2) * v(o2)

and the certain option's subjective value is ``V_c = v(certain)``. Choices
are generated from a logistic (softmax) rule with inverse-temperature
``mu``:

    P(choose gamble) = expit(mu * (V_g - V_c))

Free parameters: alpha (gain curvature), beta (loss curvature), lambda
(loss aversion), gamma (probability-weighting curvature), mu (choice
temperature).
"""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from ..utils.math import norm2alpha, norm2beta, calc_fval
from ..core.modelspec import ModelSpec


def pt_weight(p, gamma):
    """One-parameter TK1992 probability weighting function.

    ``w(p) = p**gamma / (p**gamma + (1-p)**gamma)**(1/gamma)``.
    Works elementwise on scalars or arrays. ``gamma`` is floored at a small
    epsilon so pathological optimizer excursions toward ``gamma -> 0``
    (where ``1/gamma`` blows up) can't trigger a floating-point overflow.
    """
    p = np.asarray(p, dtype=float)
    gamma = max(float(gamma), 1e-2)
    pg = np.power(p, gamma)
    qg = np.power(1.0 - p, gamma)
    return pg / np.power(pg + qg, 1.0 / gamma)


def pt_value(x, alpha, beta, lam):
    """Prospect-theory value function.

    ``v(x) = x**alpha`` for ``x >= 0``; ``v(x) = -lambda * (-x)**beta`` for
    ``x < 0``. Works elementwise on scalars or arrays (uses ``abs`` under
    the hood so fractional exponents never hit a negative base).
    """
    x = np.asarray(x, dtype=float)
    scalar_input = x.ndim == 0
    xa = np.atleast_1d(x)
    val = np.where(
        xa >= 0.0,
        np.power(np.abs(xa), alpha),
        -lam * np.power(np.abs(xa), beta),
    )
    return float(val[0]) if scalar_input else val


def _make_pt_trials(ntrials: int, rng: np.random.Generator):
    """Build a wide, well-mixed set of certain-vs-gamble trials.

    Includes gain-only, loss-only, and mixed (one gain + one loss) gambles,
    with outcome magnitudes spanning ~5-50 and probabilities spanning
    {0.1, 0.3, 0.5, 0.7, 0.9}. Certain amounts are jittered around each
    trial's objective expected value so that neither option dominates.

    Mixed (gain+loss) gambles are over-represented (roughly half of all
    trials) and about half of those pair the gamble against a near-zero
    certain amount -- the classic loss-aversion elicitation design -- since
    lambda only enters the gamble value through loss outcomes and is
    otherwise hard to disentangle from the choice temperature (mu) in
    gain-only/loss-only trials.
    """
    # over-represent mixed gambles (best source of loss-aversion information)
    group_cycle = np.array(["gain", "loss", "mixed", "mixed", "mixed"])
    groups = np.resize(group_cycle, ntrials)
    rng.shuffle(groups)

    prob_choices = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    p1 = rng.choice(prob_choices, size=ntrials)
    probs = np.stack([p1, 1.0 - p1], axis=1)

    mag = rng.uniform(5.0, 50.0, size=(ntrials, 2))
    gamble = np.zeros((ntrials, 2), dtype=float)

    gain_mask = groups == "gain"
    loss_mask = groups == "loss"
    mixed_mask = groups == "mixed"

    gamble[gain_mask, 0] = mag[gain_mask, 0]
    gamble[gain_mask, 1] = mag[gain_mask, 1]
    # make ~half of the gain (and loss) trials simple binary lotteries (X vs 0)
    zero_half = rng.random(ntrials) < 0.5
    gamble[gain_mask & zero_half, 1] = 0.0

    gamble[loss_mask, 0] = -mag[loss_mask, 0]
    gamble[loss_mask, 1] = -mag[loss_mask, 1]
    gamble[loss_mask & zero_half, 1] = 0.0

    gamble[mixed_mask, 0] = mag[mixed_mask, 0]
    gamble[mixed_mask, 1] = -mag[mixed_mask, 1]

    ev = probs[:, 0] * gamble[:, 0] + probs[:, 1] * gamble[:, 1]
    spread = 0.4 * np.maximum(np.abs(gamble[:, 0]), np.abs(gamble[:, 1]))
    certain = ev + rng.uniform(-1.0, 1.0, size=ntrials) * spread

    # for most mixed trials, pit the gamble against a near-zero certain
    # amount (classic 50/50 loss-aversion elicitation: gain X or lose Y,
    # each w.p. ~0.5, vs a sure ~0) so the choice directly probes lambda
    zero_certain = mixed_mask & (rng.random(ntrials) < 0.7)
    certain[zero_certain] = rng.uniform(-3.0, 3.0, size=int(zero_certain.sum()))
    probs[zero_certain, 0] = 0.5
    probs[zero_certain, 1] = 0.5

    return gamble, probs, certain


def pt_sim(params: np.ndarray, ntrials: int = 150, **kwargs):
    """Simulate certain-vs-gamble choices under prospect theory.

    ``params``: (nsubjects, 5) NATURAL-space values
    ``[alpha, beta, lam, gamma, mu]``.

    Returns a dict with ``gamble`` (nsubj, ntrials, 2), ``probs``
    (nsubj, ntrials, 2), ``certain`` (nsubj, ntrials), ``choice``
    (nsubj, ntrials; 1 = gamble chosen), plus diagnostics and ``params``.
    """
    params = np.asarray(params, dtype=float)
    nsubjects = params.shape[0]

    all_alpha = params[:, 0]
    all_beta = params[:, 1]
    all_lam = params[:, 2]
    all_gamma = params[:, 3]
    all_mu = params[:, 4]

    if not ((all_alpha >= 0.0) & (all_alpha <= 1.0)).all():
        raise ValueError("alpha values out of bounds [0,1]")
    if not ((all_beta >= 0.0) & (all_beta <= 1.0)).all():
        raise ValueError("beta values out of bounds [0,1]")
    if not ((all_lam >= 1e-5) & (all_lam <= 20.0)).all():
        raise ValueError("lambda values out of bounds [1e-5,20]")
    if not ((all_gamma >= 0.0) & (all_gamma <= 1.0)).all():
        raise ValueError("gamma values out of bounds [0,1]")
    if not ((all_mu >= 1e-5) & (all_mu <= 20.0)).all():
        raise ValueError("mu values out of bounds [1e-5,20]")

    rng = np.random.default_rng()

    gamble = np.zeros((nsubjects, ntrials, 2), dtype=float)
    probs = np.zeros((nsubjects, ntrials, 2), dtype=float)
    certain = np.zeros((nsubjects, ntrials), dtype=float)
    choice = np.zeros((nsubjects, ntrials), dtype=float)
    ch_prob = np.zeros((nsubjects, ntrials), dtype=float)
    Vg_all = np.zeros((nsubjects, ntrials), dtype=float)
    Vc_all = np.zeros((nsubjects, ntrials), dtype=float)

    for s in range(nsubjects):
        alpha = float(all_alpha[s])
        beta = float(all_beta[s])
        lam = float(all_lam[s])
        gamma = float(all_gamma[s])
        mu = float(all_mu[s])

        g, p, c = _make_pt_trials(ntrials, rng)
        gamble[s] = g
        probs[s] = p
        certain[s] = c

        w = pt_weight(p, gamma)  # (ntrials,2)
        v_out = pt_value(g, alpha, beta, lam)  # (ntrials,2)
        Vg = np.sum(w * v_out, axis=1)
        Vc = pt_value(c, alpha, beta, lam)

        P = expit(mu * (Vg - Vc))
        P = np.clip(P, 1e-12, 1 - 1e-12)
        ch = (rng.random(ntrials) < P).astype(float)

        ch_prob[s] = P
        Vg_all[s] = Vg
        Vc_all[s] = Vc
        choice[s] = ch

    return {
        "params": np.column_stack([all_alpha, all_beta, all_lam, all_gamma, all_mu]),
        "gamble": gamble,
        "probs": probs,
        "certain": certain,
        "choice": choice,
        "ch_prob": ch_prob,
        "Vg": Vg_all,
        "Vc": Vc_all,
    }


def pt_fit(params, gamble, probs, certain, choice, prior=None, output="npl"):
    """Thin adapter compatible with EM: returns NPL or NLL.

    ``params``: (5,) in NORMALIZED (Gaussian) space, decoded positionally as
    ``[alpha, beta, lam, gamma, mu]``.
    ``gamble``: (ntrials,2); ``probs``: (ntrials,2); ``certain``: (ntrials,);
    ``choice``: (ntrials,) with 1 = gamble chosen, 0 = certain chosen.
    """
    alpha = float(norm2alpha(params[0]))
    beta = float(norm2alpha(params[1]))
    lam = float(norm2beta(params[2]))
    gamma = float(norm2alpha(params[3]))
    mu = float(norm2beta(params[4]))

    # reject values outside natural bounds
    if not (0.0 <= alpha <= 1.0):
        return 1e7
    if not (0.0 <= beta <= 1.0):
        return 1e7
    if not (1e-5 <= lam <= 20.0):
        return 1e7
    if not (0.0 <= gamma <= 1.0):
        return 1e7
    if not (1e-5 <= mu <= 20.0):
        return 1e7

    gamble = np.asarray(gamble, dtype=float)
    probs = np.asarray(probs, dtype=float)
    certain = np.asarray(certain, dtype=float)
    choice = np.asarray(choice, dtype=float)

    w = pt_weight(probs, gamma)
    v_out = pt_value(gamble, alpha, beta, lam)
    Vg = np.sum(w * v_out, axis=1)
    Vc = pt_value(certain, alpha, beta, lam)

    P = expit(mu * (Vg - Vc))
    P = np.clip(P, 1e-12, 1 - 1e-12)
    nll = -np.sum(choice * np.log(P) + (1.0 - choice) * np.log(1.0 - P))

    if output == "all":
        return {
            "params": [alpha, beta, lam, gamma, mu],
            "gamble": gamble,
            "probs": probs,
            "certain": certain,
            "choice": choice,
            "Vg": Vg,
            "Vc": Vc,
            "ch_prob": P,
            "nll": nll,
        }

    return calc_fval(nll, params, prior=prior, output=output)


pt_desc = """Prospect Theory (Tversky & Kahneman, 1992) model of choices
between a certain amount and a two-outcome gamble. A power value function
with separate gain/loss curvature (alpha/beta) and a loss-aversion
multiplier (lambda) is combined with a one-parameter probability weighting
function (gamma); choices follow a logistic rule with temperature mu.
Free parameters: alpha, beta, lambda, gamma, mu."""
pt_id = "pt"
pt_spec = {"prospect_theory": {"value": ["alpha", "beta", "lambda"], "weighting": ["gamma"], "choice": ["mu"]}}
pt_model = ModelSpec(
    id=pt_id, spec=pt_spec, desc=pt_desc.strip(),
    params=None, sim=pt_sim, fit=pt_fit,
)
