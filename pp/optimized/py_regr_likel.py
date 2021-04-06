import ctypes
import os
import pathlib

import numpy as np


def regr_likel(
    AR_ORDER: int,
    N_SAMPLES: int,
    max_steps: int,
    theta0: np.ndarray,
    k0: float,
    xn: np.ndarray,
    eta: np.ndarray,
    wn: np.ndarray,
    xt: np.ndarray,
    wt: float,
    right_censoring: bool = False,
    do_global: bool = False,
) -> np.ndarray:
    # Convert np.array to ctype doubles pointers
    theta0 = theta0.astype(np.double)
    theta0_p = theta0.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    xn = xn.astype(np.double)
    xn_p = xn.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    eta = eta.astype(np.double)
    eta_p = eta.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    wn = wn.astype(np.double)
    wn_p = wn.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    xt = xt.astype(np.double)
    xt_p = xt.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    regr_likel_cdll.regr_likel.restype = np.ctypeslib.ndpointer(
        dtype=ctypes.c_double, shape=(AR_ORDER + 2,)
    )
    # Compute result...
    return regr_likel_cdll.regr_likel(
        AR_ORDER,
        N_SAMPLES,
        max_steps,
        theta0_p,
        k0,
        xn_p,
        eta_p,
        wn_p,
        xt_p,
        wt,
        1 if right_censoring else 0,
        1 if do_global else 0,
    )


def compute_lambda_mpfr(wt: float, mu: float, k: float) -> float:  # pragma: no cover
    return regr_likel_cdll.compute_lambda(wt, mu, k)


def ig_gradient(
    xn: np.ndarray, thetap: np.ndarray, k: float, wn: np.ndarray, eta: np.ndarray
) -> np.ndarray:

    N_SAMPLES = xn.shape[0]
    AR_ORDER = xn.shape[1] - 1

    grad = np.zeros(AR_ORDER + 2)
    grad = grad.astype(np.double)
    grad_p = grad.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    # compute mus
    mus = np.dot(xn, thetap)
    # convert to pointers
    xn = xn.astype(np.double)
    xn_p = xn.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    mus = mus.astype(np.double)
    mus_p = mus.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    wn = wn.astype(np.double)
    wn_p = wn.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    eta = eta.astype(np.double)
    eta_p = eta.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    regr_likel_cdll.ig_gradient(
        N_SAMPLES, AR_ORDER, xn_p, mus_p, wn_p, k, eta_p, grad_p
    )
    return grad


def mpfr_ig_gradient_rc(
    k: float, thetap: np.ndarray, xt: np.ndarray, wt: float, rc_eta: float
) -> np.ndarray:
    assert (
        xt.shape[0] == 1 and xt.shape[1] > xt.shape[0]
    )  # xt should be a column vector
    # compute rc_mu
    rc_mu = np.dot(xt, thetap.reshape(-1, 1))[0, 0]
    AR_ORDER = xt.shape[1] - 1
    rc_grad = np.zeros(AR_ORDER + 2)
    rc_grad = rc_grad.astype(np.double)
    rc_grad_p = rc_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    xt = xt.astype(np.double)
    xt_p = xt.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    regr_likel_cdll.test_mpfr_ig_gradient_rc(
        AR_ORDER, k, wt, rc_mu, rc_eta, xt_p, rc_grad_p
    )
    return rc_grad


abs_path = pathlib.Path(__file__).parent.absolute()

# FIXME not really safe
so_file = os.path.join(abs_path, "regr_likel.so")
c_file = os.path.join(abs_path, "c_regr_likel.c")

if not os.path.isfile(so_file):  # pragma: no cover
    print(f"It appears that {so_file} is not present\nCompiling {c_file}...")
    # On Mac and Linux
    os.system(f"gcc -lnlopt -lmpfr -lgmp -fPIC -shared -o {so_file} {c_file}")
    print("Compilation completed! \U0001F389\U0001F389")

regr_likel_cdll = ctypes.CDLL(so_file)

regr_likel_cdll.regr_likel.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_int,
]
regr_likel_cdll.compute_lambda.argtypes = [
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
]

regr_likel_cdll.compute_lambda.restype = ctypes.c_double

regr_likel_cdll.ig_gradient.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]


regr_likel_cdll.test_mpfr_ig_gradient_rc.argtypes = [
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]
