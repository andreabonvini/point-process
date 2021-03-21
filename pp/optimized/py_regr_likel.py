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
) -> np.ndarray:
    # Convert np.array to ctype doubles
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
        AR_ORDER, N_SAMPLES, max_steps, theta0_p, k0, xn_p, eta_p, wn_p, xt_p, wt
    )


# so_file generated with:
# cc -lnlopt -fPIC -shared -o regr_likel.so c_regr_likel.c

abs_path = pathlib.Path(__file__).parent.absolute()

so_file = os.path.join(abs_path, "regr_likel.so")
c_file = os.path.join(abs_path, "c_regr_likel.c")

if not os.path.isfile(so_file):  # pragma: no cover
    print(f"It appears that {so_file} is not present\nCompiling {c_file}...")
    # On Mac
    os.system(f"cc -lnlopt -fPIC -shared -o {so_file} {c_file}")
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
]
