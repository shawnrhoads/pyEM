---
name: pyem-model-generator
description: Generate new computational cognitive model modules and example notebooks for pyEM-style workflows, including from free-text task/model descriptions. Use this skill to scaffold model-class shared utilities (`modclass_utils.py`), per-model files with `mod_desc/mod_spec/mod_id/MODEL`, and parameter-recovery notebooks when adding models not present in base packages.
---

# pyem-model-generator

Generate standalone model code and notebooks from bundled references only.

## Required resources (always local to this skill)

- `references/template.json`
- `references/rl.json`
- `references/bayes.json`
- `references/glm.json`
- `references/modelclass-utils-template.py`
- `references/model-file-template.py`
- `references/example-notebook-template.json`
- `references/parameter-recovery-notebook.md`
- `references/pyem-runtime-contract.md`

Do not assume repository model files or installed pyem package are available.

## Clarification behavior

If required information is missing, ask concise follow-up questions before generation.

Required confirmations:

1. `model_class`, `model_name`, and output paths.
2. Task structure (`nsubjects`, `nblocks`, `ntrials`, choice count).
3. Parameter names, transforms, bounds, and priors.
4. Update equations and choice rule.
5. Variant definitions and naming.
6. Notebook requirements (recovery metrics and plots).

## Free-text parsing workflow

When the user gives prose/equations instead of structured JSON:

1. Place original text in `description_input.raw_text`.
2. Parse into `description_input.extracted_spec`:
   - task flow and outcomes,
   - tensor shapes,
   - update equations,
   - choice rule(s),
   - variant rules.
3. Normalize symbols into valid Python names.
4. Ask targeted questions for ambiguities (sign conventions, variant toggles, data keys).
5. Preserve equation intent when generating `mod_sim` and `mod_fit`.

## Model-class utility heuristic (required)

Generate shared utility module first, then model files:

- Shared module: `pyem/models/{model_class}_utils.py`
- Model module(s): `pyem/models/{model_name}.py`

Each model file must import shared helpers using this contract:

```python
from .modclass_utils import _alloc_sim, _alloc_fit, ModelSpec, spec_to_id, build_params
```

Shared helper expectations:

- `_alloc_sim` / `_alloc_fit`: tensor allocation.
- `ModelSpec`: model metadata registration.
- `spec_to_id`: deterministic model ID from `mod_spec`.
- `build_params`: parameter initialization and transforms.

## Per-model file contract

Each generated `{model_name}.py` should include:

- attributes: `mod_desc`, `mod_spec`, `mod_id`, `MODEL`
- functions: `mod_params`, `mod_sim`, `mod_fit`

Use `references/model-file-template.py` as the base pattern.

## Notebook generation contract

Generate `examples/{model_class}.ipynb` from `references/example-notebook-template.json`.

Required notebook flow:

1. model/task overview markdown
2. parameter setup
3. simulation run
4. fit/recovery run
5. parameter recovery plots (identity line + correlation)

See `references/parameter-recovery-notebook.md` for section and plotting details.

## Generation steps

1. Load `references/template.json`.
2. Merge user inputs (or parsed free-text spec) into template fields.
3. Generate `modclass_utils.py` from `references/modelclass-utils-template.py`.
4. Generate `{model_name}.py` from `references/model-file-template.py` with required attributes/functions.
5. Generate notebook from `references/example-notebook-template.json`.
6. Run smoke checks on generated code/notebook cells when execution is requested.

## Smoke checks

- Import generated utils and model modules.
- Run `mod_params`, `mod_sim`, and `mod_fit(output="npl")` on minimal synthetic data.
- Verify notebook cells execute through recovery plotting.
