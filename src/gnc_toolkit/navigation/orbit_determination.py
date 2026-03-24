"""
Extended Kalman Filter for Orbit Determination (OD-EKF).
"""

import numpy as np

from gnc_toolkit.disturbances.gravity import J2Gravity, TwoBodyGravity
from gnc_toolkit.kalman_filters.ekf import EKF


class OrbitDeterminationEKF:
    """
    Extended Kalman Filter for Orbit Determination (OD-EKF).
    Estimates position and velocity in ECI frame.
    """

    def __init__(self, x0, P0, Q, R, use_j2=True, mu=398600.4418e9, re=6378137.0):
        """
        Initialize the OD-EKF.

        Args:
            x0 (np.ndarray): Initial state [r_x, r_y, r_z, v_x, v_y, v_z]
            P0 (np.ndarray): Initial covariance (6x6)
            Q (np.ndarray): Process noise covariance (6x6)
            R (np.ndarray): Measurement noise covariance (3x3)
            use_j2 (bool): Whether to use J2 gravity model
            mu (float): Gravitational parameter (m^3/s^2)
            re (float): Earth radius (m)
        """
        self.ekf = EKF(dim_x=6, dim_z=3)
        self.ekf.x = x0.astype(float)
        self.ekf.P = P0.astype(float)
        self.ekf.Q = Q.astype(float)
        self.ekf.R = R.astype(float)

        self.mu = mu
        self.re = re
        self.use_j2 = use_j2

        if use_j2:
            self.gravity = J2Gravity(mu=mu, re=re)
        else:
            self.gravity = TwoBodyGravity(mu=mu)

    def _state_transition(self, x, dt, u=None, **kwargs):
        """
        State transition function: x_dot = f(x)
        Using 4th order Runge-Kutta for integration.
        """

        def f(state):
            r = state[:3]
            v = state[3:]
            a = self.gravity.get_acceleration(r)
            return np.concatenate([v, a])

        # RK4 Integration
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)

        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _jacobian_f(self, x, dt, u=None, **kwargs):
        """
        Jacobian of state transition function.
        F = I + Phi * dt
        """
        r = x[:3]
        r_mag = np.linalg.norm(r)

        # Two-body gravity gradient
        G = (self.mu / r_mag**3) * (3.0 * np.outer(r, r) / r_mag**2 - np.eye(3))

        Phi = np.zeros((6, 6))
        Phi[:3, 3:] = np.eye(3)
        Phi[3:, :3] = G

        # First-order approximation
        F = np.eye(6) + Phi * dt
        return F

    def predict(self, dt):
        """Perform EKF prediction step."""
        self.ekf.predict(self._state_transition, self._jacobian_f, dt)

    def update(self, z_pos):
        """
        Perform EKF update step with position measurement.
        z_pos: [x, y, z] measurement in ECI
        """

        def hx(x):
            return x[:3]

        def H_jac(x):
            H = np.zeros((3, 6))
            H[:, :3] = np.eye(3)
            return H

        self.ekf.update(z_pos, hx, H_jac)

    @property
    def state(self):
        return self.ekf.x

    @property
    def covariance(self):
        return self.ekf.P
