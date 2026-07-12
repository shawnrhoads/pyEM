import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

seaborn = pytest.importorskip("seaborn")
from matplotlib.axes import Axes
from pyem.utils import plotting


def test_plot_scatter_returns_axes():
    x = np.linspace(0, 1, 20)
    y = x + np.random.default_rng(0).normal(scale=0.1, size=20)
    ax = plotting.plot_scatter(x, "true", y, "est", show_line=True)
    assert isinstance(ax, Axes)
