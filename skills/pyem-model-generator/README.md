# pyem-model-generator skill

Use this skill to scaffold new computational cognitive models for pyEM that are not in base `pyem`, including from free-text task/model descriptions with equations.

## What it generates

- `pyem/models/{model_class}.py`
  - `{model_name}_sim(params, nblocks, ntrials, **kwargs)`
  - `{model_name}_fit(params, *, prior=None, output="npl", **kwargs)`
- `examples/{model_class}.ipynb`
  - model/task description
  - simulation and fit demo
  - parameter recovery plot (like `examples/rl.ipynb`)

## Included templates

- `template.json` (simple main template)
- `rl.json`, `bayes.json`, `glm.json` (reference anchors)
- `references/example-notebook-template.json` (parameter recovery notebook template)

## How to use

1. Copy and fill `template.json`.
2. Provide it to the skill in your prompt.
3. Answer follow-up questions if anything is missing.
4. Ask the skill to generate the model module and notebook.


## Offline resources

If the runtime does not include full `pyem` source files, use:

- `references/pyem-runtime-contract.md`
- `references/parameter-recovery-notebook.md`

These provide enough contract detail to generate pyEM-compatible sim/fit functions and notebook recovery plots.


## Free-text description support

If you provide prose + equations instead of a filled template, the skill will parse text into `description_input.extracted_spec`.
Use `description_examples.social_signals` in `template.json` as a worked example of this conversion.
