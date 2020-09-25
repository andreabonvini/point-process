from unittest import TestCase
from unittest.mock import patch

import numpy as np

from pp import utils


class TestUtils(TestCase):
    xn = np.array(
        [
            [1.0, 0.953, 1.0, 0.992, 0.984, 0.985, 0.906, 0.844, 0.875, 0.898],
            [1.0, 0.86, 0.953, 1.0, 0.992, 0.984, 0.985, 0.906, 0.844, 0.875],
            [1.0, 0.859, 0.86, 0.953, 1.0, 0.992, 0.984, 0.985, 0.906, 0.844],
            [1.0, 0.875, 0.859, 0.86, 0.953, 1.0, 0.992, 0.984, 0.985, 0.906],
            [1.0, 0.891, 0.875, 0.859, 0.86, 0.953, 1.0, 0.992, 0.984, 0.985],
            [1.0, 0.859, 0.891, 0.875, 0.859, 0.86, 0.953, 1.0, 0.992, 0.984],
        ]
    )

    m, n = xn.shape
    wn = np.ones((m, 1))
    eta = np.ones((m, 1))
    thetap = np.ones((n, 1))
    params = np.vstack([0.5, eta, thetap]).squeeze(1)

    def test_load(self):
        path = "tests/test_data/Y2.csv"
        r = utils.load(path)
        assert r.shape == (733,)

    @patch("pp.utils.plt.plot")
    def test_plot_intervals(self, plot):
        plot.return_value = True
        r = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        utils.plot_intervals(r)
        plot.assert_called_once()

    def test_unpack_invgauss_params(self):
        res = utils.unpack_invgauss_params(self.params, self.m, self.n)
        assert type(res[0]) == np.float64
        assert res[1].shape == (self.m, 1)
        assert res[2].shape == (self.n, 1)

    def test_likel_invgauss_consistency_check_good(self):
        res = utils.likel_invgauss_consistency_check(
            xn=self.xn, wn=self.wn, eta=self.eta, xt=None, thetap0=self.thetap
        )
        self.assertIsNone(res)

    def test_likel_invgauss_consistency_check_wn_bad(self):
        with self.assertRaises(ValueError):
            utils.likel_invgauss_consistency_check(
                xn=self.xn,
                wn=np.ones((self.m - 1, 1)),
                eta=self.eta,
                xt=None,
                thetap0=self.thetap,
            )

    def test_likel_invgauss_consistency_check_xt_bad(self):
        with self.assertRaises(ValueError):
            utils.likel_invgauss_consistency_check(
                xn=self.xn,
                wn=self.wn,
                eta=self.eta,
                xt=np.ones((1, self.n - 1)),
                thetap0=self.thetap,
            )

    def test_likel_invgauss_consistency_check_eta_bad(self):
        with self.assertRaises(ValueError):
            utils.likel_invgauss_consistency_check(
                xn=self.xn,
                wn=self.wn,
                eta=np.ones((self.m - 1, 1)),
                xt=None,
                thetap0=self.thetap,
            )

    def test_likel_invgauss_consistency_check_thetap_bad(self):
        with self.assertRaises(ValueError):
            utils.likel_invgauss_consistency_check(
                xn=self.xn,
                wn=self.wn,
                eta=self.eta,
                xt=None,
                thetap0=np.ones((self.n - 1, 1)),
            )
