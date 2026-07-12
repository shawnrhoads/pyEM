# Installation

## Requirements

pyEM requires **Python ≥ 3.10** and the following runtime dependencies (installed automatically):

- `numpy>=1.22`
- `scipy>=1.10`
- `pandas>=1.5`
- `matplotlib`
- `joblib>=1.3`
- `typing-extensions>=4.6`

## Install from GitHub

pyEM is not currently on PyPI. Install it directly from a clone of the GitHub repository:

```bash
git clone https://github.com/shawnrhoads/pyEM.git
cd pyEM
pip install -e .
```

## Optional extras

Some functionality (plotting helpers, additional model-fitting utilities) is gated behind optional extras so the base install stays lightweight:

```bash
# seaborn, for plotting helpers such as plot_recovery()
pip install -e ".[viz]"

# statsmodels, scikit-learn, tqdm — used by some model/analysis utilities
pip install -e ".[extras]"
```

## Documentation dependencies

To build this documentation site locally, install the docs requirements:

```bash
pip install -r requirements-docs.txt
```

## Verify your install

Run the following snippet to confirm pyEM imports correctly and a built-in model is available:

```python
import pyem
from pyem import EMModel, FitResult
from pyem.models.rl_mf import rw1a1b_model
print("pyEM import OK:", EMModel.__name__, rw1a1b_model.id)
```

This should print `pyEM import OK: EMModel rw1a1b` with no errors.
