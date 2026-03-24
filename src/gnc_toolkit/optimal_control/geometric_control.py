"""
Geometric Controller on SO(3) for attitude tracking.
"""

import numpy as np


def vee_map(R):
    """
    Vee map (skew-symmetric extractor).
    Converts a skew-symmetric matrix to a vector in R^3.
    """
    return np.array([R[2, 1], R[0, 2], R[1, 0]])


def hat_map(v):
    """
    Hat map (skew-symmetric matrix generator).
    Converts a vector in R^3 to a skew-symmetric matrix.
    """
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class GeometricController:
    """
    Geometric Controller on SO(3) for attitude tracking.
    Based on Lee et al. (2010), "Geometric Numerical Integration of Differential Equations".
    Provides global stability and avoids singularities.
    """

    def __init__(self, J, kR, kW):
        """
        Initialize the Geometric Controller.

        Args:
            J (np.ndarray): Inertia tensor matrix [3x3].
            kR (float): Proportional gain for attitude error.
            kW (float): Derivative gain for angular velocity error.
        """
        self.J = np.array(J)
        self.kR = float(kR)
        self.kW = float(kW)

    def compute_control(self, R, omega, R_d, omega_d, d_omega_d=None):
        """
        Compute control torque.

        Args:
            R (np.ndarray): Current rotation matrix [3x3].
            omega (np.ndarray): Current angular velocity [3].
            R_d (np.ndarray): Desired rotation matrix [3x3].
            omega_d (np.ndarray): Desired angular velocity [3].
            d_omega_d (np.ndarray, optional): Desired angular acceleration [3]. Defaults to zeros.

        Returns
        -------
            np.ndarray: Control torque [3].
        """
        R = np.array(R)
        omega = np.array(omega)
        R_d = np.array(R_d)
        omega_d = np.array(omega_d)
        if d_omega_d is None:
            d_omega_d = np.zeros(3)
        else:
            d_omega_d = np.array(d_omega_d)

        # Attitude error: eR = 0.5 * vee(R_d.T @ R - R.T @ R_d)
        error_matrix = R_d.T @ R - R.T @ R_d
        eR = 0.5 * vee_map(error_matrix)

        # Angular velocity error: eW = omega - R.T @ R_d @ omega_d
        R_rel = R.T @ R_d
        eW = omega - R_rel @ omega_d

        # Feedforward: gyroscopic and inertia-weighted acceleration terms
        gyroscopic = np.cross(omega, self.J @ omega)
        omega_skew = hat_map(omega)
        acc_term = omega_skew @ R_rel @ omega_d - R_rel @ d_omega_d
        feedforward = self.J @ acc_term

        M = -self.kR * eR - self.kW * eW + gyroscopic - feedforward

        return M
