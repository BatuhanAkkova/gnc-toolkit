"""
Attitude guidance laws for Nadir, Sun, and Target pointing, and trajectory planning.
"""

import numpy as np

from gnc_toolkit.utils.quat_utils import axis_angle_to_quat, quat_normalize


def nadir_pointing_reference(pos_eci: np.ndarray, vel_eci: np.ndarray) -> np.ndarray:
    """
    Generate a reference quaternion for Nadir pointing.
    Target frame: Z-axis points towards Earth center, Y-axis along orbit normal,
    X-axis completes the right-handed frame (approximately velocity direction).

    Args:
        pos_eci: ECI position vector [m] (3,)
        vel_eci: ECI velocity vector [m/s] (3,)

    Returns
    -------
        Reference quaternion [x, y, z, w]
    """
    pos_eci = np.asarray(pos_eci)
    vel_eci = np.asarray(vel_eci)

    # Z-axis (Nadir): Towards Earth center
    z_axis = -pos_eci / np.linalg.norm(pos_eci)

    # Y-axis: Negative of Orbit normal (to align with negative Z_eci for equatorial)
    orb_normal = np.cross(pos_eci, vel_eci)
    y_axis = -orb_normal / np.linalg.norm(orb_normal)

    # X-axis: Complete the frame (approximates velocity direction in circular orbit)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # DCM from ECI to Body (R_eb)
    rmat = np.array([x_axis, y_axis, z_axis])

    return _rmat_to_quat(rmat)


def sun_pointing_reference(sun_vec_eci: np.ndarray) -> np.ndarray:
    """
    Generate a reference quaternion for Sun pointing.
    Primary objective: Align Body X-axis with Sun vector.
    Secondary objective: Minimize rotation about Sun vector (constrained pointing).

    Args:
        sun_vec_eci: ECI Sun direction vector (3,)

    Returns
    -------
        Reference quaternion [x, y, z, w]
    """
    s = sun_vec_eci / np.linalg.norm(sun_vec_eci)

    # Simple algorithm: find shortest rotation from ECI X to Sun vector
    eci_x = np.array([1.0, 0.0, 0.0])

    dot = np.dot(eci_x, s)
    if dot > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0])
    if dot < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0])

    axis = np.cross(eci_x, s)
    angle = np.arccos(dot)

    return axis_angle_to_quat(axis / np.linalg.norm(axis) * angle)


def target_tracking_reference(pos_eci: np.ndarray, target_pos_eci: np.ndarray) -> np.ndarray:
    """
    Generate a reference quaternion to track a target position.
    Target frame: Z-axis aligned with boresight (towards target).

    Args:
        pos_eci: ECI position vector of spacecraft [m] (3,)
        target_pos_eci: ECI position vector of target [m] (3,)

    Returns
    -------
        Reference quaternion [x, y, z, w]
    """
    rel_pos = target_pos_eci - pos_eci
    z_axis = rel_pos / np.linalg.norm(rel_pos)

    eci_z = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z_axis, eci_z)) > 0.99:
        eci_z = np.array([1.0, 0.0, 0.0])

    y_axis = np.cross(z_axis, eci_z)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)

    rmat = np.vstack([x_axis, y_axis, z_axis])
    return _rmat_to_quat(rmat)


def eigenaxis_slew_path_planning(
    q_initial: np.ndarray, q_final: np.ndarray, time_span: np.ndarray
) -> list[np.ndarray]:
    """
    Generate an eigenaxis slew profile between two orientations.
    Using SLERP (Spherical Linear Interpolation).

    Args:
        q_initial: Initial quaternion [x, y, z, w]
        q_final: Final quaternion [x, y, z, w]
        time_span: Normalized time values [0, 1] (N,)

    Returns
    -------
        List of quaternions along the path
    """
    path = []
    for t in time_span:
        path.append(attitude_blending(q_initial, q_final, t))
    return path


def attitude_blending(q1: np.ndarray, q2: np.ndarray, alpha: float) -> np.ndarray:
    """
    SLERP (Spherical Linear Interpolation) between two quaternions.

    Args:
        q1: First quaternion [x, y, z, w]
        q2: Second quaternion [x, y, z, w]
        alpha: Blending factor [0, 1]

    Returns
    -------
        Blended quaternion [x, y, z, w]
    """
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)

    dot = np.dot(q1, q2)

    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        res = q1 + alpha * (q2 - q1)
        return quat_normalize(res)

    theta_0 = np.arccos(dot)
    theta = theta_0 * alpha

    sin_theta_0 = np.sin(theta_0)
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (s0 * q1) + (s1 * q2)


def _rmat_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Internal helper to convert 3x3 rotation matrix to quaternion [x, y, z, w].
    Robust implementation (Shepperd's algorithm).
    """
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    return np.array([x, y, z, w])
