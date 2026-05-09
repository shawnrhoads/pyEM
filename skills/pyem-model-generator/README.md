# pyem-model-generator skill

Use this skill to generate standalone computational cognitive model code and a matching parameter-recovery notebook.

## What this skill generates

Given a task/model description, the skill generates files in a **single directory**:

- `{modclass}_utils.py`
- `{model_name}.py`
- `{model_name}.ipynb`

The generated model file follows a consistent contract:

- attributes: `mod_desc`, `mod_spec`, `mod_id`, `MODEL`
- functions: `mod_params`, `mod_sim`, `mod_fit`

## Required references bundled with the skill

- `references/rl.json`
- `references/bayes.json`
- `references/glm.json`
- `references/modelclass-utils-template.py`
- `references/model-file-template.py`
- `references/example-notebook-template.json`
- `references/parameter-recovery-notebook.md`
- `references/pyem-runtime-contract.md`

## Quick start (first-time users)

1. Describe your task and model in plain language (or equations).
2. Ask the skill to generate:
   - `{modclass}_utils.py`
   - `{model_name}.py`
   - `{model_name}.ipynb`
3. If details are missing, answer the skill’s follow-up questions.
4. Review generated files and run your analysis workflow.

## Notes on generated files

### Shared utils file

`{modclass}_utils.py` should define shared helpers used across model files:

- `_alloc_sim`, `_alloc_fit`
- `ModelSpec`, `ParamDef`
- `spec_to_id`, `build_params`
- `PARAM_REGISTRY`

### Model file

`{model_name}.py` imports math helpers from pyEM:

```python
from pyem.utils.math import norm2alpha, norm2beta, softmax, calc_fval
```

And imports shared helpers from:

```python
from {modclass}_utils import _alloc_sim, _alloc_fit, ModelSpec, spec_to_id, build_params
```

### Notebook file

The notebook template uses:

```python
from pyem.api import EMModel
```

and follows a simulation → fit → recovery plot workflow.

## Example prompt

```text
Use pyem-model-generator.
Generate standalone files in one directory:
- social_utils.py
- social_rw.py
- social_rw.ipynb

Task: three-option social learning task with 4 blocks x 12 trials and 100 agents.
Model: dual-value update equations for self and other values with softmax choice.
Include parameter recovery plots in the notebook.
Ask follow-up questions before generation if any details are ambiguous.
```
