import numpy as np
from pyem.core.posterior import parameter_recovery


def test_parameter_recovery_function():
    true = np.array([[0.0, 1.0], [1.0, 0.0]])
    est = true + 0.1
    res = parameter_recovery(true, est)
    assert res.corr.shape == (2,)
    assert np.allclose(res.rmse, 0.1, atol=1e-6)


def test_plotting_has_no_toplevel_seaborn():
    import ast
    import pathlib
    import pyem.utils.plotting as p
    src = pathlib.Path(p.__file__).read_text()
    tree = ast.parse(src)
    toplevel = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [a.name for n in toplevel if isinstance(n, ast.Import) for a in n.names]
    names += [n.module for n in toplevel if isinstance(n, ast.ImportFrom) and n.module]
    assert not any(n == "seaborn" or n.startswith("seaborn.") for n in names)
