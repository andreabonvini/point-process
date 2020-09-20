from unittest.mock import patch

import numpy as np

from pp.utils import load, plot_intervals


def test_load():
    path = "tests/test_data/Y2.csv"
    r = load(path)
    assert r.shape == (733,)


@patch("pp.utils.plt.plot")
def test_plot_intervals(plot):
    plot.return_value = True
    r = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    plot_intervals(r)
    plot.assert_called_once()
