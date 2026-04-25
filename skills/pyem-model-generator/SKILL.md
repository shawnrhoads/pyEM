---
name: pyem-model-generator
description: Generate new computational cognitive model modules for pyEM and matching example notebooks. Use when asked to add a model not included in base pyEM, scaffold `pyem.models.{modelclass}.py` with `{modelname}_sim(params, nblocks, ntrials, **kwargs)` and `{modelname}_fit(params, *, prior=None, output="npl")`, and produce `examples/{modelclass}.ipynb`. Trigger this skill when the user wants pyEM-style imports, parameter transformations (e.g., `norm2alpha`, `norm2beta`), output dictionaries, model variants, or RL/Bayes/GLM-aligned structure.
---

# pyem-model-generator

Generate pyEM-compatible model code using patterns in:

- `pyem/models/rl.py`
- `pyem/models/bayes.py`
- `pyem/models/glm.py`

## Offline/resource mode

When full `pyem` package files are unavailable, load:

- `references/pyem-runtime-contract.md` for utility and fit contracts.
- `references/parameter-recovery-notebook.md` for notebook structure and plotting requirements.

In offline mode, follow these references instead of guessing utility behavior.


## Mandatory clarification behavior

If any required information is missing or ambiguous, ask the user concise follow-up questions before generating code.

Required items to confirm:

1. `model_class`, `model_name`, and target module path.
2. Simulation task inputs (at minimum `nblocks` and `ntrials`; plus task-specific arrays).
3. Parameter list with transform/bounds and semantic role.
4. Sim output keys and fit output modes.
5. Variant definitions (if requested).

## pyEM import and function format (pseudo code)

```python
import numpy as np
from ..utils.math import softmax, norm2alpha, norm2beta, calc_fval


def {model_name}_sim(
    params: np.ndarray,
    nblocks: int = 4,
    ntrials: int = 12,
    **kwargs,
) -> dict:
    """Simulate behavior for one model family."""
    n_subjects = params.shape[0]
    rng = np.random.default_rng(kwargs.get("seed", None))

    beta = params[:, 0]   # natural-space for simulation
    alpha = params[:, 1]  # natural-space for simulation

    choices = np.empty((n_subjects, nblocks, ntrials), dtype=object)
    rewards = np.zeros((n_subjects, nblocks, ntrials), dtype=float)
    ev = np.zeros((n_subjects, nblocks, ntrials + 1, 2), dtype=float)
    pe = np.zeros((n_subjects, nblocks, ntrials), dtype=float)
    nll = np.zeros((n_subjects, nblocks, ntrials), dtype=float)

    for s in range(n_subjects):
        for b in range(nblocks):
            ev[s, b, 0, :] = 0.5
            for t in range(ntrials):
                p = softmax(ev[s, b, t, :], beta[s])
                c = rng.choice([0, 1], p=p)
                r = 0.0  # replace with task-specific outcome logic
                pe[s, b, t] = r - ev[s, b, t, c]
                ev[s, b, t + 1, :] = ev[s, b, t, :]
                ev[s, b, t + 1, c] = ev[s, b, t, c] + alpha[s] * pe[s, b, t]
                nll[s, b, t] = -np.log(p[c] + 1e-12)

    return {
        "params": params,
        "choices": choices,
        "rewards": rewards,
        "EV": ev,
        "PE": pe,
        "nll": nll,
    }


def {model_name}_fit(
    params,
    *,
    prior=None,
    output: str = "npl",
    **kwargs,
):
    """Compute fit objective compatible with pyEM."""
    beta = float(norm2beta(params[0]))
    alpha = float(norm2alpha(params[1]))

    if not (1e-5 <= beta <= 20.0):
        return 1e7
    if not (0.0 <= alpha <= 1.0):
        return 1e7

    choices = kwargs["choices"]
    rewards = kwargs["rewards"]
    nblocks, ntrials = rewards.shape

    ev = np.zeros((nblocks, ntrials + 1, 2), dtype=float)
    pe = np.zeros((nblocks, ntrials), dtype=float)
    nll = 0.0

    for b in range(nblocks):
        ev[b, 0, :] = 0.5
        for t in range(ntrials):
            c = 0 if choices[b, t] == "A" else 1
            p = softmax(ev[b, t, :], beta)
            r = rewards[b, t]
            pe[b, t] = r - ev[b, t, c]
            ev[b, t + 1, :] = ev[b, t, :]
            ev[b, t + 1, c] = ev[b, t, c] + alpha * pe[b, t]
            nll += -np.log(p[c] + 1e-12)

    if output == "all":
        return {"params": np.array([beta, alpha]), "EV": ev, "PE": pe, "nll": nll}

    return calc_fval(nll, params, prior=prior, output=output)
```

## Generation workflow

1. Load `template.json` and, if package context is missing, load the two files under `references/`.
2. If fields are missing, ask follow-up questions and wait for answers.
3. Generate `pyem/models/{model_class}.py` using imports, signatures, transforms, and output keys in the template.
4. Generate `examples/{model_class}.ipynb` with:
   - model/task description,
   - simulation demo,
   - fit simulated behavior,
   - parameter recovery plot (same intent as `examples/rl.ipynb`).
5. Run smoke checks: import module, run sim, run fit with `output="npl"`.

## Optional reference alignment

If needed, consult `rl.json`, `bayes.json`, and `glm.json` to mirror existing style and output contracts.
