# References

- Source code: [github.com/shawnrhoads/pyEM](https://github.com/shawnrhoads/pyEM)
- Back to [Home](index.md)

## Related methods & papers

The model families included in pyEM are implementations of (or are directly inspired by) the following methods and papers.

**Model-Free Reinforcement Learning**  (`pyem/models/rl_mf.py`)
- Sutton, R. S., & Barto, A. G. (1998). Reinforcement Learning: An Introduction.
- Rescorla, R. A. (1972). A theory of Pavlovian conditioning: Variations in the effectiveness of reinforcement and non-reinforcement. *Classical conditioning, Current Research and Theory*, 2, 64-69.

**Model-Free Prosocial Reinforcement Learning**  (`pyem/models/rl_mf.py`)
- Lockwood, P. L., Apps, M. A., Valton, V., Viding, E., & Roiser, J. P. (2016). Neurocomputational mechanisms of prosocial learning and links to empathy. Proceedings of the National Academy of Sciences, 113(35), 9763-9768. https://doi.org/10.1073/pnas.1603198113
- Rhoads, S. A., Gan, L., Berluti, K., O’Connell, K., Cutler, J., Lockwood, P. L., & Marsh, A. A. (2025). Neurocomputational basis of learning when choices simultaneously affect both oneself and others. Nature Communications, 16(1), 9350. https://doi.org/10.1038/s41467-025-64424-9

**Model-Based Reinforcement Learning**  (`pyem/models/rl_mb.py`)
- Daw, N. D., Gershman, S. J., Seymour, B., Dayan, P., & Dolan, R. J. (2011). Model-based influences on humans' choices and striatal prediction errors. *Neuron*, 69(6), 1204-1215. https://doi.org/10.1016/j.neuron.2011.02.027

**Signal Detection Theory** (`pyem/models/sdt.py`)
- Green, D. M. (1966). Signal detection theory and psychophysics.
- Lockhart, R. S., & Murdock, B. B. (1970). Memory and the theory of signal detection. *Psychological Bulletin*, 74(2), 100–109. https://doi.org/10.1037/h0029536

**Prospect Theory** (`pyem/models/pt.py`)
- Tversky, A., & Kahneman, D. (1992). Advances in prospect theory: Cumulative representation of uncertainty. *Journal of Risk and Uncertainty*, 5(4), 297-323. https://doi.org/10.1007/BF00122574

**Social Discounting** (`pyem/models/discounting.py`)
- Jones, B., & Rachlin, H. (2006). Social discounting. *Psychological Science*, 17(4), 283-286.
- Rhoads, S. A., Vekaria, K. M., O'Connell, K., Elizabeth, H. S., Rand, D. G., Kozak Williams, M. N., & Marsh, A. A. (2023). Unselfish traits and social decision-making patterns characterize six populations of real-world extraordinary altruists. *Nature Communications*, 14(1), 1807. https://doi.org/10.1038/s41467-023-37283-5
- Rhoads, S. A., O'Connell, K., Berluti, K., Ploe, M. L., Elizabeth, H. S., Amormino, P., Li, J. L., Dutton, M., VanMeter, A. S., & Marsh, A. A. (2023). Neural responses underlying extraordinary altruists' generosity for socially distant others. *PNAS Nexus*, 2(7), pgad199. https://doi.org/10.1093/pnasnexus/pgad199

**Temporal Discounting** (`pyem/models/discounting.py`)
- Mazur, J. E. (1987). An adjusting procedure for studying delayed reinforcement. In M. L. Commons, J. E. Mazur, J. A. Nevin, & H. Rachlin (Eds.), The effect of delay and of intervening events on reinforcement value (pp. 55–73).

**Probability Discounting** (`pyem/models/discounting.py`)
- Rachlin, H., Raineri, A., & Cross, D. (1991). Subjective probability and delay. Journal of the Experimental Analysis of Behavior, 55(2), 233-244. https://doi.org/10.1901/jeab.1991.55-233

**Effort Discounting** (`pyem/models/discounting.py`)
- Hartmann, M. N., Hager, O. M., Tobler, P. N. & Kaiser, S. Parabolic discounting of monetary rewards by physical effort. *Behavioural Processes*. 100, 192–196 (2013). https://doi.org/10.1016/j.beproc.2013.09.014

**Prosocial Effort Discounting** (`pyem/models/discounting.py`)
- Lockwood, P. L., Hamonet, M., Zhang, S. H., Ratnavel, A., Salmony, F. U., Husain, M., & Apps, M. A. (2017). Prosocial apathy for helping others when effort is required. *Nature Human Behaviour*, 1(7), 0131. https://doi.org/10.1038/s41562-017-0131

**Drift Diffusion** (`pyem/models/ddm.py`)
- Navarro, D. J., & Fuss, I. G. (2009). Fast and accurate calculations for first-passage times in Wiener diffusion models. Journal of mathematical psychology, 53(4), 222-230. https://doi.org/10.1016/j.jmp.2009.02.003
- Ratcliff, R., & Tuerlinckx, F. (2002). Estimating parameters of the diffusion model: Approaches to dealing with contaminant reaction times and parameter variability. *Psychonomic Bulletin & Review*, 9(3), 438-481. https://doi.org/10.3758/BF03196302

**Bayesian Inference** (`pyem/models/bayes.py`)
- Fiore, V. G., & Gu, X. (2022). Similar network compositions, but distinct neural dynamics underlying belief updating in environments with and without explicit outcomes. *NeuroImage*, 247, 118821. https://doi.org/10.1016/j.neuroimage.2021.118821
- Fiore, V., Kiely, A., Zhang, E., Simonetti, J., Phadnis, A., Yang, S., Berner, L., & Smith, A. (2026). Widely used bandit tasks elicit diverging belief updating phenotypes in healthy adults. *Research Square Preprint*. https://doi.org/10.21203/rs.3.rs-9439488/v1

**Expectation Maximization with Maximum a Posteriori**
- Daw, N. D. (2011). Trial-by-trial data analysis using computational models. *Decision making, affect, and learning: Attention and performance XXIII*, 23(1). https://doi.org/10.1093/acprof:oso/9780199600434.003.0001 [<a href="https://www.princeton.edu/~ndaw/d10.pdf">pdf</a>]
- Huys, Q. J., Cools, R., Gölzer, M., Friedel, E., Heinz, A., Dolan, R. J., & Dayan, P. (2011). Disentangling the roles of approach, activation and valence in instrumental and pavlovian responding. *PLoS Computational Biology*, 7(4), e1002028. https://doi.org/10.1371/journal.pcbi.1002028
- Wittmann, M. K., Fouragnan, E., Folloni, D., Klein-Flügge, M. C., Chau, B. K., Khamassi, M., & Rushworth, M. F. (2020). Global reward state affects learning and activity in raphe nucleus and anterior insula in monkeys. *Nature Communications*, 11(1), 3771. https://doi.org/10.1038/s41467-020-17343-w
- Cutler, J., Wittmann, M. K., Abdurahman, A., Hargitai, L. D., Drew, D., Husain, M., & Lockwood, P. L. (2021). Ageing is associated with disrupted reinforcement learning whilst learning to help others is preserved. *Nature Communications*, 12(1), 4440. https://doi.org/10.1038/s41467-021-24576-w
- Rhoads, S. A., Gan, L., Berluti, K., O'Connell, K., Cutler, J., Lockwood, P. L., & Marsh, A. A. (2025). Neurocomputational basis of learning when choices simultaneously affect both oneself and others. *Nature Communications*. 16, 9350. https://doi.org/10.1038/s41467-025-64424-9

**Papers using pyEM**
- Rhoads, S., Bhattacharyya, K., Kulkarni, K., & Gu, X. (2026). Social expectations and homeostatic dynamics shape momentary social craving. *PsyArXiv*. https://osf.io/preprints/psyarxiv/yjdre
- Kulkarni, K. R., Berner, L. A., Rhoads, S. A., Fiore, V. G., Schiller, D., & Gu, X. (2026). A computational mechanism linking momentary craving and decision-making in alcohol drinkers and cannabis users. *Nature Mental Health*, 4, 551–565 (2026). https://doi.org/10.1038/s44220-026-00593-w
- Fu, Q. X., Shevlin, B. R., Davis, A. N., Heflin, M., Kulkarni, K. R., Saez, I., Mayberg, H. S., & Gu, X. (2026). Context-specificity and longitudinal stability of the effects of depressive symptoms on social decision-making. *PsyArXiv*. https://doi.org/10.31234/osf.io/rtda4_v1