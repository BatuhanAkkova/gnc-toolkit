import numpy as np

def quat_normalize(q):
    """Return the normalized quaternion [x, y, z, w]."""
    norm = quat_norm(q)
    if norm == 0:
        raise ValueError("Quaternion norm is zero.")
    return q / norm

def quat_conj(q):
    """Conjugate of a quaternion [x, y, z, w]."""
    return np.array([-q[0], -q[1], -q[2], q[3]])

def quat_norm(q):
    """Return the norm of the quaternion [x, y, z, w]."""
    return np.linalg.norm(q)

def quat_inv(q):
    """Return the inverse of the quaternion [x, y, z, w]."""
    norm_sq = np.sum(q**2)
    if norm_sq == 0:
        raise ValueError("Quaternion norm is zero.")
    return quat_conj(q) / norm_sq

def quat_mult(q1, q2):
    """Return the product of two quaternions q1 * q2 using [x, y, z, w] convention."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 + y1*w2 + z1*x2 - x1*z2,
        w1*z2 + z1*w2 + x1*y2 - y1*x2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def quat_rot(q, v):
    """Rotate a vector v by quaternion q using q * [v, 0] * q_inv."""
    # Ensure v is a 3-element vector
    v = np.asarray(v)
    v_quat = np.array([v[0], v[1], v[2], 0.0])
    
    # q_rot = q * v_quat * q_conj
    res = quat_mult(quat_mult(q, v_quat), quat_conj(q))
    return res[0:3]

def quat_to_rmat(q):
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
    ])

def axis_angle_to_quat(axis_angle):
    """Convert axis-angle to quaternion."""
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-10:
        return np.array([0, 0, 0, 1.0])
    axis = axis_angle / angle
    half_angle = angle / 2
    return np.array([
        axis[0] * np.sin(half_angle),
        axis[1] * np.sin(half_angle),
        axis[2] * np.sin(half_angle),
        np.cos(half_angle)
    ])

def skew_symmetric(v):
    """Return the skew-symmetric matrix of vector v."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
