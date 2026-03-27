"""
Lander/Rover surface navigation EKF using landmark tracking.
"""

import numpy as np
from typing import Optional

from gnc_toolkit.kalman_filters.ekf import EKF


class SurfaceNavigationEKF:
    r"""
    Surface Navigation EKF for Landers or Rovers.

    Estimates 6D state $\mathbf{x} = [\mathbf{r}, \mathbf{v}]^T$ in a local 
    surface-fixed frame using constant acceleration kinematics and landmark 
    observations.

    Parameters
    ----------
    x0 : np.ndarray
        Initial state $[x, y, z, v_x, v_y, v_z]^T$ (m, m/s).
    p0 : np.ndarray
        Initial estimation error covariance ($6\times 6$).
    q_mat : np.ndarray
        Process noise covariance ($6\times 6$).
    r_mat : np.ndarray
        Measurement noise covariance for relative landmark tracking ($3\times 3$).
    """

    def __init__(self, x0: np.ndarray, p0: np.ndarray, q_mat: np.ndarray, r_mat: np.ndarray):
        """Initialize Surface EKF."""
        self.ekf = EKF(dim_x=6, dim_z=3)
        self.ekf.x = np.asarray(x0, dtype=float)
        self.ekf.P = np.asarray(p0, dtype=float)
        self.ekf.Q = np.asarray(q_mat, dtype=float)
        self.ekf.R = np.asarray(r_mat, dtype=float)

    def predict(self, dt: float, accel: Optional[np.ndarray] = None) -> None:
        """
        Perform kinematic state prediction.

        Parameters
        ----------
        dt : float
            Propagation step (s).
        accel : np.ndarray, optional
            IMU acceleration or commanded thrust (m/s^2).
        """
        def fx(x: np.ndarray, dt_step: float, u: Optional[np.ndarray]) -> np.ndarray:
            r, v = x[:3], x[3:]
            a = np.asarray(u) if u is not None else np.zeros(3)
            return np.concatenate([r + v*dt_step + 0.5*a*dt_step**2, v + a*dt_step])

        def f_jac(x: np.ndarray, dt_step: float, u: Optional[np.ndarray]) -> np.ndarray:
            phi = np.eye(6)
            phi[:3, 3:] = np.eye(3) * dt_step
            return phi

        self.ekf.predict(fx, f_jac, dt, u=accel)

    def update_landmark(self, z_obs: np.ndarray, landmark_pos: np.ndarray) -> None:
        r"""
        Update state using an observed relative vector to a known landmark.

        Measurement model: $\mathbf{z} = \mathbf{r}_{land} - \mathbf{r}_{Rover} + \nu$.

        Parameters
        ----------
        z_obs : np.ndarray
            Measured relative vector (m).
        landmark_pos : np.ndarray
            Known coordinates of the landmark (m).
        """
        r_land = np.asarray(landmark_pos)

        def hx(x: np.ndarray) -> np.ndarray:
            return r_land - x[:3]

        def h_jac(x_state_unused: np.ndarray) -> np.ndarray:
            h = np.zeros((3, 6))
            h[:, :3] = -np.eye(3)
            return h

        self.ekf.update(np.asarray(z_obs), hx, h_jac)

    @property
    def state(self) -> np.ndarray:
        """Estimated 6D rover state vector."""
        return self.ekf.x
