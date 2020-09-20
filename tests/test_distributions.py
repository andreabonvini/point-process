from unittest import TestCase

import numpy as np

from pp.distributions import inverse_gaussian


class TestInverseGaussian(TestCase):
    def test_inverse_gaussian_floats(self):
        res = inverse_gaussian(xs=2.0, mus=2.0, lamb=1.0)
        assert 1 > res > 0

    def test_inverse_gaussian_vectors(self):
        xs = np.array([1, 2, 3, 4, 5]).astype(float).reshape((5, 1))
        mus = np.array([1, 2, 3, 4, 5]).astype(float).reshape((5, 1))
        res = inverse_gaussian(xs=xs, mus=mus, lamb=1.0)
        assert res.shape == (5, 1) and all(1 > p > 0 for p in res)

    def test_inverse_gaussian_value_error(self):
        xs = np.array([1, 2, 3, 4, 5]).astype(float).reshape((5, 1))
        mus = np.array([1, 2, 3, 4]).astype(float).reshape((4, 1))
        with self.assertRaises(ValueError):
            inverse_gaussian(xs=xs, mus=mus, lamb=1.0)

    def test_inverse_gaussian_type_error(self):
        xs = np.array([1, 2, 3, 4, 5]).astype(float).reshape((5, 1))
        mus = 0.5
        with self.assertRaises(TypeError):
            inverse_gaussian(xs=xs, mus=mus, lamb=1.0)
