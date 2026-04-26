# pyem-model-generator skill

Use this skill to scaffold standalone computational cognitive model files and recovery notebooks from reference specs or free-text equations.

## Plan to fix known flaws

1. Remove smoke-test requirements from the skill workflow.
2. Enforce notebook import of `EMModel` (`from pyem.api import EMModel`) and forbid `scipy.optimize.minimize` usage in notebook templates.
3. Remove duplicated template files (`template.json`, `references/template.json`).
4. Enforce flat output layout in one directory (no `pyem/models/...` or `examples/...` paths).
5. Restrict shared utils to exactly: `_alloc_sim`, `_alloc_fit`, `ModelSpec`, `ParamDef`, `spec_to_id`, `build_params`, `PARAM_REGISTRY`.
6. Keep math helper imports (`norm2alpha`, `norm2beta`, `softmax`, `calc_fval`) in model files and document them in `references/pyem-runtime-contract.md`.

## Current references

- `references/rl.json`
- `references/bayes.json`
- `references/glm.json`
- `references/modelclass-utils-template.py`
- `references/model-file-template.py`
- `references/example-notebook-template.json`
- `references/parameter-recovery-notebook.md`
- `references/pyem-runtime-contract.md`

## Output contract

Generate all files in the same directory:

- `modclass_utils.py`
- `{model_name}.py`
- `{model_name}.ipynb`

Each `{model_name}.py` must define:

- `mod_desc`, `mod_spec`, `mod_id`, `MODEL`
- `mod_params`, `mod_sim`, `mod_fit`

Each notebook must use:

```python
from pyem.api import EMModel
```
