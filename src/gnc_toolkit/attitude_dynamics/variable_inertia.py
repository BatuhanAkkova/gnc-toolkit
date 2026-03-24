"""
Attitude dynamics with time-varying inertia tensors.
"""

import numpy as np


def variable_inertia_euler_equations(J, J_dot, omega, torque):
    """
    Computes angular acceleration for a body with time-varying inertia.

    The equation of motion is:
    J * omega_dot + J_dot * omega + omega x (J * omega) = torque

    Args:
        J (np.ndarray): Inertia tensor (3, 3) [kg*m^2].
        J_dot (np.ndarray): Time derivative of inertia tensor (3, 3) [kg*m^2/s].
        omega (np.ndarray): Angular velocity vector (3,) [rad/s].
        torque (np.ndarray): External torque vector (3,) [N*m].

    Returns
    -------
        np.ndarray: Angular acceleration vector (3,) [rad/s^2].
    """
    # Angular momentum: H = J * omega
    H = J @ omega

    # Gyroscopic term: omega x H
    gyro_term = np.cross(omega, H)

    # Inertia change term: J_dot * omega
    inertia_change_term = J_dot @ omega

    # RHS for solving J * omega_dot = torque - inertia_change - gyro
    rhs = torque - inertia_change_term - gyro_term

    # Solve for angular acceleration
    omega_dot = np.linalg.solve(J, rhs)

    return omega_dot


def mass_depletion_J_dot(J_nominal, m_initial, dm_dt, r_point):
    """
    A simple model for J_dot due to point-mass depletion (e.g., fuel at r_point).

    J = J_rigid + m(t) * [ (r'r)I - rr' ]
    J_dot = m_dot * [ (r'r)I - rr' ]

    Args:
        J_nominal (np.ndarray): Nominal inertia (unused here, placeholder).
        m_initial (float): Initial mass of segment [kg].
        dm_dt (float): Mass flow rate [kg/s] (usually negative).
        r_point (np.ndarray): Position of mass segment from CM (3,).

    Returns
    -------
        np.ndarray: J_dot (3, 3).
    """
    r_sq = np.dot(r_point, r_point)
    r_outer = np.outer(r_point, r_point)

    J_dot = dm_dt * (r_sq * np.eye(3) - r_outer)

    return J_dot
