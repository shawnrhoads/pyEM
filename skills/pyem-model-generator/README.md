# pyem-model-generator skill

Use this skill to scaffold new computational cognitive model modules and example notebooks from either structured JSON or free-text equations.

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
