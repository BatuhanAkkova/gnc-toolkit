"""
Cayley-Klein parameters for attitude representation and composition.
"""

import numpy as np
from typing import cast


def quat_to_cayley_klein(q: np.ndarray) -> np.ndarray:
    r"""
    Convert a quaternion to Cayley-Klein parameters.

    The Cayley-Klein parameters $(\alpha, \beta)$ are complex numbers
    representing the rotation:
    $\alpha = w + i z$
    $\beta = y + i x$

    Parameters
    ----------
    q : np.ndarray
        Quaternion [x, y, z, w].

    Returns
    -------
    np.ndarray
        2x2 complex unitary matrix:
        [[alpha, beta],
         [-beta*, alpha*]]
    """
    qv = np.asarray(q)
    x, y, z, w = qv
    alpha = complex(w, z)
    beta = complex(y, x)

    return cast(np.ndarray, np.array([
        [alpha, beta],
        [-np.conj(beta), np.conj(alpha)]
    ]))


def cayley_klein_to_quat(u_mat: np.ndarray) -> np.ndarray:
    """
    Convert a 2x2 Cayley-Klein matrix to a quaternion.

    Parameters
    ----------
    u_mat : np.ndarray
        2x2 complex Cayley-Klein matrix.

    Returns
    -------
    np.ndarray
        Quaternion [x, y, z, w].
    """
    umat = np.asarray(u_mat)
    alpha = umat[0, 0]
    beta = umat[0, 1]

    w = alpha.real
    z = alpha.imag
    y = beta.real
    x = beta.imag

    return cast(np.ndarray, np.array([x, y, z, w]))


def cayley_klein_mult(u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    """
    Multiply two Cayley-Klein matrices (composes rotations).

    Parameters
    ----------
    u1 : np.ndarray
        First 2x2 complex matrix.
    u2 : np.ndarray
        Second 2x2 complex matrix.

    Returns
    -------
    np.ndarray
        Product 2x2 complex matrix.
    """
    return cast(np.ndarray, np.asarray(u1) @ np.asarray(u2))




