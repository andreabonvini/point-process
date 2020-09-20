from unittest import TestCase

import numpy as np

from pp.likelihood import compute_loglikel, likel_invnorm


def test_compute_loglikel():
    eta = np.ones((5, 1))
    k = 0.5
    mus = np.ones((5, 1))
    wn = np.ones((5, 1))
    res = compute_loglikel(eta, k, mus, wn)
    assert isinstance(res, np.float64)


class TestLikelInvnorm(TestCase):
    def test_likel_invnorm(self):
        xn = np.ones((5, 5))
        wn = np.ones((5, 1))
        none = likel_invnorm(xn, wn)
        # TODO actually implement some test...
        assert none is None
