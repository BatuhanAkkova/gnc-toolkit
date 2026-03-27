"""
Quaternion kinematics and math utilities.
"""


import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    r"""
    Normalize a quaternion to unit length.

    Parameters
    ----------
    q : np.ndarray
        Quaternion [x, y, z, w].

    Returns
    -------
    np.ndarray
        Unit quaternion $\mathbf{q} / \|\mathbf{q}\|$.
    """
    qv = np.asarray(q)
    norm = np.linalg.norm(qv)
    if norm < 1e-15:
        raise ValueError("Cannot normalize a zero-length quaternion.")
    return qv / norm


def quat_conj(q: np.ndarray) -> np.ndarray:
    """
    Compute the conjugate of a quaternion.

    Parameters
    ----------
    q : np.ndarray
        Quaternion [x, y, z, w].

    Returns
    -------
    np.ndarray
        Conjugate quaternion [-x, -y, -z, w].
    """
    qv = np.asarray(q)
    return np.array([-qv[0], -qv[1], -qv[2], qv[3]])


def quat_norm(q: np.ndarray) -> float:
    """
    Compute the norm of a quaternion.

    Parameters
    ----------
    q : np.ndarray
        Quaternion [x, y, z, w].

    Returns
    -------
    float
        Norm of the quaternion.
    """
    qv = np.asarray(q)
    return np.linalg.norm(qv)


def quat_mult(q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
    r"""
    Multiply two quaternions (Hamilton product).

    Equation:
    $\mathbf{q}_c = \mathbf{q}_a \otimes \mathbf{q}_b$

    Parameters
    ----------
    q_left : np.ndarray
        Left quaternion $[x_1, y_1, z_1, w_1]$.
    q_right : np.ndarray
        Right quaternion $[x_2, y_2, z_2, w_2]$.

    Returns
    -------
    np.ndarray
        Resulting quaternion $[x_c, y_c, z_c, w_c]$.
    """
    ql = np.asarray(q_left)
    qr = np.asarray(q_right)

    x1, y1, z1, w1 = ql
    x2, y2, z2, w2 = qr

    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
        w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def quat_inv(q: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a quaternion.

    Parameters
    ----------
    q : np.ndarray
        Quaternion [x, y, z, w].

    Returns
    -------
    np.ndarray
        Inverse quaternion.
    """
    qv = np.asarray(q)
    norm = quat_norm(qv)
    if norm < 1e-15:
        raise ValueError("Cannot invert a zero-length quaternion.")
    return quat_conj(qv) / norm**2


def quat_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    r"""
    Rotate a 3D vector by a quaternion.

    Equation:
    $\mathbf{v}' = \mathbf{q} \otimes [0, \mathbf{v}] \otimes \mathbf{q}^*$

    Parameters
    ----------
    q : np.ndarray
        Unit quaternion $[x, y, z, w]$.
    v : np.ndarray
        Vector to rotate $[x, y, z]$.

    Returns
    -------
    np.ndarray
        Rotated vector $[x', y', z']$.
    """
    qv = np.asarray(q)
    vv = np.asarray(v)

    v_ext = np.array([vv[0], vv[1], vv[2], 0.0])
    q_inv = quat_conj(qv)

    res = quat_mult(quat_mult(qv, v_ext), q_inv)
    return res[0:3]


def quat_to_rmat(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to a 3x3 Direction Cosine Matrix (DCM).

    Parameters
    ----------
    q : np.ndarray
        Unit quaternion [x, y, z, w].

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    qv = np.asarray(q)
    x, y, z, w = qv

    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
    ])


def axis_angle_to_quat(axis: np.ndarray, angle: float | None = None) -> np.ndarray:
    r"""
    Convert axis and angle (or a rotation vector) to a rotation quaternion.

    Parameters
    ----------
    axis : np.ndarray
        Rotation axis $[u_x, u_y, u_z]$ (if angle is provided) 
        or rotation vector $\boldsymbol{\theta}$ (if angle is None).
    angle : float, optional
        Rotation angle (radians).

    Returns
    -------
    np.ndarray
        Unit quaternion [x, y, z, w].
    """
    av = np.asarray(axis)

    if angle is None:
        norm = np.linalg.norm(av)
        if norm < 1e-15:
            return np.array([0.0, 0.0, 0.0, 1.0])
        u = av / norm
        theta = norm
    else:
        norm = np.linalg.norm(av)
        if norm < 1e-15:
            return np.array([0.0, 0.0, 0.0, 1.0])
        u = av / norm
        theta = angle

    s = np.sin(theta / 2.0)
    c = np.cos(theta / 2.0)

    return np.array([u[0] * s, u[1] * s, u[2] * s, c])


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    r"""
    Create a 3x3 skew-symmetric matrix from a vector.

    Used for cross products: $a \times b = [a]_\times b$.

    Parameters
    ----------
    v : np.ndarray
        3D vector.

    Returns
    -------
    np.ndarray
        3x3 skew-symmetric matrix.
    """
    vv = np.asarray(v)
    return np.array([
        [0.0,   -vv[2],  vv[1]],
        [vv[2],  0.0,   -vv[0]],
        [-vv[1], vv[0],  0.0]
    ])




