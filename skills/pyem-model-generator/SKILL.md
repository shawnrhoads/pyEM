---
name: pyem-model-generator
description: Generate standalone computational cognitive model modules and example notebooks from free-text or reference specs, using a shared `modclass_utils.py` contract and per-model files with `mod_desc`, `mod_spec`, `mod_id`, `MODEL`, `mod_params`, `mod_sim`, and `mod_fit`.
---

# pyem-model-generator

Generate all outputs into the **current working directory** (flat layout).

## Required local references

- `references/rl.json`
- `references/bayes.json`
- `references/glm.json`
- `references/modelclass-utils-template.py`
- `references/model-file-template.py`
- `references/example-notebook-template.json`
- `references/parameter-recovery-notebook.md`
- `references/pyem-runtime-contract.md`

Do not require repository path conventions like `pyem/models/...` or `examples/...`.

## Output layout (flat)

Write files in one directory:

- `{modclass}_utils.py`
- `{model_name}.py` (one or more model files)
- `{model_name}.ipynb` (or one notebook per model class)

## Clarification behavior

If required details are missing, ask concise follow-up questions before generation:

1. Task structure (`nsubjects`, `nblocks`, `ntrials`, choices, outcomes).
2. Parameter names/transforms/bounds/priors.
3. Equations (state update and choice rule).
4. Variant list and naming.
5. Desired output filenames.

## Free-text parsing workflow

When given prose/equations:

1. Extract task flow, tensors, equations, and variants.
2. Normalize symbol names to valid Python variables.
3. Preserve equation intent in `mod_sim`/`mod_fit`.
4. Resolve ambiguities via targeted questions.

## Shared utility heuristic (required)

Create one shared `{modclass}_utils.py` file containing only:

- `_alloc_sim`
- `_alloc_fit`
- `ModelSpec`
- `ParamDef`
- `spec_to_id`
- `build_params`
- `PARAM_REGISTRY`

Each `{model_name}.py` should import shared helpers with:

```python
from {modclass}_utils import _alloc_sim, _alloc_fit, ModelSpec, spec_to_id, build_params
```

## Per-model file contract

Each generated `{model_name}.py` must include:

- attributes: `mod_desc`, `mod_spec`, `mod_id`, `MODEL`
- functions: `mod_params`, `mod_sim`, `mod_fit`

Each model file should import math helpers directly from pyem:

```python
from pyem.utils.math import norm2alpha, norm2beta, softmax, calc_fval
```

## Notebook requirements

Generate notebook from `references/example-notebook-template.json` and ensure it imports:

```python
from pyem.api import EMModel
```

Do not use `from scipy.optimize import minimize` in generated notebooks.

## Generation steps

1. Select the closest anchor from `references/rl.json`, `references/bayes.json`, `references/glm.json`.
2. Generate `modclass_utils.py` from `references/modelclass-utils-template.py`.
3. Generate each `{model_name}.py` from `references/model-file-template.py`.
4. Generate notebook(s) from `references/example-notebook-template.json` and `references/parameter-recovery-notebook.md`.
