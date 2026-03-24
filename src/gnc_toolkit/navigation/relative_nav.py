"""
EKF for relative navigation using Clohessy-Wiltshire (Hill) dynamics.
"""

import numpy as np

from gnc_toolkit.kalman_filters.ekf import EKF


class RelativeNavigationEKF:
    """
    EKF for relative navigation using Clohessy-Wiltshire (Hill) dynamics.
    Estimates relative position and velocity [x, y, z, vx, vy, vz].
    """

    def __init__(self, x0, P0, Q, R, n):
        """
        Args:
            x0 (np.ndarray): Initial relative state [m, m/s].
            P0 (np.ndarray): Initial covariance.
            Q (np.ndarray): Process noise covariance.
            R (np.ndarray): Measurement noise covariance (3x3 for rel pos).
            n (float): Mean motion of the target orbit (rad/s).
        """
        self.ekf = EKF(dim_x=6, dim_z=3)
        self.ekf.x = x0.astype(float)
        self.ekf.P = P0.astype(float)
        self.ekf.Q = Q.astype(float)
        self.ekf.R = R.astype(float)
        self.n = n

    def predict(self, dt):
        """Predict relative state using CW equations."""
        n = self.n

        def fx(x, dt, u):
            # State vector: [x, y, z, vx, vy, vz]
            # Use discrete-time CW propagation
            F = self._get_cw_transition_matrix(n, dt)
            return F @ x

        def F_jac(x, dt, u):
            return self._get_cw_transition_matrix(n, dt)

        self.ekf.predict(fx, F_jac, dt)

    def update(self, z_rel_pos):
        """Update relative state using relative position measurement."""

        def hx(x):
            return x[:3]  # Position only measurement

        def H_jac(x):
            H = np.zeros((3, 6))
            H[:, :3] = np.eye(3)
            return H

        self.ekf.update(z_rel_pos, hx, H_jac)

    def _get_cw_transition_matrix(self, n, t):
        """Analytical Clohessy-Wiltshire State Transition Matrix."""
        nt = n * t
        s = np.sin(nt)
        c = np.cos(nt)

        F = np.zeros((6, 6))

        # Phi_rr (Position from Position)
        F[0, 0] = 4 - 3 * c
        F[1, 0] = 6 * (s - nt)
        F[1, 1] = 1
        F[2, 2] = c

        # Phi_rv (Position from Velocity)
        F[0, 3] = s / n
        F[0, 4] = (2 / n) * (1 - c)
        F[1, 3] = (2 / n) * (c - 1)
        F[1, 4] = (4 * s / n) - 3 * t
        F[2, 5] = s / n

        # Phi_vr (Velocity from Position)
        F[3, 0] = 3 * n * s
        F[4, 0] = 6 * n * (c - 1)
        F[5, 2] = -n * s

        # Phi_vv (Velocity from Velocity)
        F[3, 3] = c
        F[3, 4] = 2 * s
        F[4, 3] = -2 * s
        F[4, 4] = 4 * c - 3
        F[5, 5] = c

        return F

    @property
    def state(self):
        return self.ekf.x
