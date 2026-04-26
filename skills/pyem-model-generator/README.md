# pyem-model-generator skill

Use this skill to scaffold new computational cognitive model modules and example notebooks from either structured JSON or free-text equations.

## Review summary

This skill is currently in good shape for offline generation because it has:

- A canonical, local template (`references/template.json`) and a convenience copy (`template.json`).
- Local reference anchors for RL/Bayes/GLM (`references/rl.json`, `references/bayes.json`, `references/glm.json`).
- Self-contained model scaffolding templates (`references/modelclass-utils-template.py`, `references/model-file-template.py`) with shared helper contracts.
- Notebook generation templates and parameter-recovery guidance (`references/example-notebook-template.json`, `references/parameter-recovery-notebook.md`).
- Runtime math/objective contracts in `references/pyem-runtime-contract.md`.

## Primary template

- `references/template.json` (canonical template)
- `template.json` (copy of canonical template for convenience)

## Reference anchors

- `references/rl.json`
- `references/bayes.json`
- `references/glm.json`

## Model-class utility layout

Generated model classes should include:

- `pyem/models/{model_class}_utils.py`
- one or more `pyem/models/{model_name}.py`

Each model file should follow this shared import contract:

```python
from .modclass_utils import _alloc_sim, _alloc_fit, ModelSpec, spec_to_id, build_params
```

Each generated `{model_name}.py` should define:

- attributes: `mod_desc`, `mod_spec`, `mod_id`, `MODEL`
- functions: `mod_params`, `mod_sim`, `mod_fit`

## Offline resources

This skill is self-contained and does not require repository model files:

- `references/modelclass-utils-template.py`
- `references/model-file-template.py`
- `references/example-notebook-template.json`
- `references/parameter-recovery-notebook.md`
- `references/pyem-runtime-contract.md`

## Example prompts

Use prompts like the following with this skill.

### 1) Reversal learning (Kalman filter RL)

```text
Use pyem-model-generator.
Task: Reversal learning RL with two options A (80% reward) and B (20% reward), 2 blocks, 40 trials per block, reversals every 10 trials.
Per trial: choose A or B, then observe reward (+1) or no reward (0).
Generate a Kalman filter model to simulate behavior and fit the same model to simulated behavior.
Please output:
1) pyem/models/{model_class}_utils.py
2) pyem/models/{model_name}.py with mod_desc/mod_spec/mod_id/MODEL and mod_params/mod_sim/mod_fit
3) examples/{model_class}.ipynb with parameter recovery plots.
If anything is ambiguous, ask follow-up questions first.
```

### 2) Two-step task (model-free, model-based, hybrid)

```text
Use pyem-model-generator.
Task: Two-step task with two stages per trial.
Stage 1: choose between two first-stage options; common transition p=0.70 and rare transition p=0.30 to one of two second-stage states (fixed mapping).
Stage 2: choose between two options in reached state; reward/no reward outcome.
Second-stage reward probabilities (4 arms) drift by independent bounded Gaussian random walks in [0.25, 0.75].
Generate three variants:
- Model-free SARSA(lambda) with parameters alpha_1, alpha_2, lambda, beta_1, beta_2, p
- Model-based with Bellman-style prospective first-stage valuation
- Hybrid with Q_net = w*Q_MB + (1-w)*Q_TD and parameters beta_1, beta_2, alpha_1, alpha_2, lambda, p, w
Please generate files using the model-class utility layout and include a recovery notebook that compares recovered parameters for all variants.
Ask clarifying questions if needed.
```

### 3) Social signals task with variants

```text
Use pyem-model-generator.
The task is called the social signals task. On each trial, participants see three options (A,B,C) and choose one option (stored in subject, block, trial arrays), then observe clear signal (+1) or not (0), then receive social feedback (thumbs up/down).
Use 100 agents, 4 blocks, 12 trials per block.
Model equations:
Q_self[s,b,t+1,c] = Q_self[s,b,t,c] + alpha_self * (outcome_self[s,b,t] - Q_self[s,b,t,c])
Q_other[s,b,t+1,c] = Q_other[s,b,t,c] + alpha_other * (outcome_other[s,b,t] - Q_other[s,b,t,c])
outcome_self in {0,1}
outcome_other = social_sensitivity_pos*1 (if positive) or social_sensitivity_neg*-1 (if negative), label this theta
p(choice) = softmax(beta * (w_self*Q_self[s,b,t,c] + w_other*Q_other[s,b,t,c]))
Variants include:
- 1b2w2a
- 1b2w2a2t
- 1b2w2a4t
- arbitration variants: 1b1o2a4t, 1b1o1a4t, 1b1o2a2t, 1b1o2a using
  p(choice) = (1-omega)*softmax(beta * Q_self[s,b,t,c]) + omega*softmax(beta * Q_other[s,b,t,c])
Generate utility + model files + notebook with parameter recovery for each variant.
Ask follow-up questions for any ambiguous naming/sign conventions.
```
