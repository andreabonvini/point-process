import numpy as np
from scipy.optimize import LinearConstraint


def greater_than_zero(n_var: int):
    A = np.identity(n_var)
    lb = np.zeros((n_var,))
    ub = np.ones((n_var,)) * np.inf
    return LinearConstraint(A, lb, ub)
