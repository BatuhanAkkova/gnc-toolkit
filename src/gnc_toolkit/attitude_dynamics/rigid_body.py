"""
Rigid body Attitude dynamics based on Euler's equations of motion.
"""

import numpy as np

def euler_equations(J: np.ndarray, omega: np.ndarray, torque: np.ndarray) -> np.ndarray:
    """
    Computes the angular acceleration of a rigid body using Euler's equations of motion.

    Args:
        J (np.ndarray): Inertia tensor (3x3 matrix) [kg*m^2].
        omega (np.ndarray): Angular velocity vector (3,) [rad/s].
        torque (np.ndarray): External torque vector (3,) [N*m].

    Returns:
        np.ndarray: Angular acceleration vector (3,) [rad/s^2].
    
    Raises:
        ValueError: If input shapes are incorrect.
    """
    # Input validation
    if J.shape != (3, 3):
        raise ValueError(f"Inertia tensor J must be shape (3, 3), got {J.shape}")
    if omega.shape != (3,):
        raise ValueError(f"Angular velocity omega must be shape (3,), got {omega.shape}")
    if torque.shape != (3,):
        raise ValueError(f"Torque vector must be shape (3,), got {torque.shape}")

    # Euler's equations: J * omega_dot + omega x (J * omega) = torque
    # omega_dot = J_inv * (torque - omega x (J * omega))
    
    # Calculate angular momentum
    H = J @ omega
    
    # Calculate gyroscopic term (omega x H)
    gyro_term = np.cross(omega, H)
    
    # Calculate the right hand side (torque - gyro_term)
    rhs = torque - gyro_term
    
    # Solve for angular acceleration (omega_dot)
    omega_dot = np.linalg.solve(J, rhs)
    
    return omega_dot
