# References

- Source code: [github.com/shawnrhoads/pyEM](https://github.com/shawnrhoads/pyEM)
- Back to [Home](index.md)

## Related methods & papers

The model families included in pyEM are implementations of (or are directly inspired by) the following methods and papers, as cited in the corresponding model docstrings. Venue/page details are included only where given in the source; entries without a venue in the docstring are marked `[uncertain]` rather than guessed.

- Daw, N. D., Gershman, S. J., Seymour, B., Dayan, P., & Dolan, R. J. (2011). Model-based influences on humans' choices and striatal prediction errors. *Neuron*, 69, 1204–1215. — two-step task; model-based reinforcement learning (`pyem/models/rl_mb.py`).
- Lockwood, P. L., Apps, M. A. J., Valton, V., Viding, E., & Roiser, J. P. (2016). *[uncertain — venue not given in source]* — three-learning-rate (3α) reinforcement-learning model (`pyem/models/rl_mf.py`).
- Rhoads, S. A. et al. (2025). *[uncertain — venue not given in source]* — four-learning-rate (4α) reinforcement-learning model distinguishing recipient (self/other) and valence (positive/negative) (`pyem/models/rl_mf.py`).
- Tversky, A., & Kahneman, D. (1992). *[uncertain — venue not given in source]* — cumulative prospect theory model of choices (`pyem/models/pt.py`).
- Mazur, J. E. (1987). *[uncertain — venue not given in source]* — hyperbolic temporal (delay) discounting (`pyem/models/discounting.py`).
- Rachlin, H., Raineri, A., & Cross, D. (1991). *[uncertain — venue not given in source]* — hyperbolic probability discounting (`pyem/models/discounting.py`).
- Prévost, C., Pessiglione, M., Météreau, E., Cléry-Melin, M.-L., & Dreher, J.-C. (2010); Hartmann, M. N. et al. (2013). *[uncertain — venue not given in source]* — parabolic (accelerating-cost) effort discounting (`pyem/models/discounting.py`).
- Lockwood, P. L., Apps, M. A. J., Valton, V., Viding, E., & Roiser, J. P. (2017). *[uncertain — venue not given in source]* — self/other prosocial effort discounting paradigm (`pyem/models/discounting.py`).
- Navarro, D. J., & Fuss, I. G. (2009). *Journal of Mathematical Psychology*, 53, 222–230. — Wiener first-passage-time (WFPT) likelihood for drift-diffusion models (`pyem/models/ddm.py`).
- Ratcliff, R., & Tuerlinckx, F. (2002). *Psychonomic Bulletin & Review*, 9, 438–481. — drift-diffusion models with trial-to-trial variability (`pyem/models/ddm.py`).
