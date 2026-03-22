"""
Cayley-Klein parameters for attitude representation and composition.
"""

import numpy as np

def quat_to_cayley_klein(q):
    """
    Convert quaternion [x, y, z, w] to Cayley-Klein parameters (alpha, beta).
    
    alpha = w + i*z
    beta = y + i*x
    
    Returns a 2x2 complex matrix:
    [[alpha, beta],
     [-beta*, alpha*]]
    """
    x, y, z, w = q
    alpha = complex(w, z)
    beta = complex(y, x)
    
    return np.array([
        [alpha, beta],
        [-np.conj(beta), np.conj(alpha)]
    ])

def cayley_klein_to_quat(U):
    """
    Convert 2x2 Cayley-Klein matrix to quaternion [x, y, z, w].
    """
    alpha = U[0, 0]
    beta = U[0, 1]
    
    w = alpha.real
    z = alpha.imag
    y = beta.real
    x = beta.imag
    
    return np.array([x, y, z, w])

def cayley_klein_mult(U1, U2):
    """
    Multiply two Cayley-Klein matrices (composes rotations).
    """
    return U1 @ U2
