import numpy as np
from gnc_toolkit.kalman_filters.ekf import EKF

class SurfaceNavigationEKF:
    """
    Basic SLAM-like Surface Navigation EKF.
    Estimates lander/rover position and velocity [x, y, z, vx, vy, vz]
    using measurements to known landmarks.
    """
    def __init__(self, x0, P0, Q, R):
        """
        Args:
            x0 (np.ndarray): Initial state in local frame [m, m/s].
            P0 (np.ndarray): Initial covariance.
            Q (np.ndarray): Process noise covariance.
            R (np.ndarray): Measurement noise covariance (3x3 for relative vector).
        """
        self.ekf = EKF(dim_x=6, dim_z=3)
        self.ekf.x = x0.astype(float)
        self.ekf.P = P0.astype(float)
        self.ekf.Q = Q.astype(float)
        self.ekf.R = R.astype(float)

    def predict(self, dt, accel=None):
        """
        Predict state using constant velocity or constant acceleration model.
        
        Args:
            dt (float): Time step [s].
            accel (np.ndarray, optional): Constant acceleration input [m/s^2].
        """
        def fx(x, dt, u):
            r = x[:3]
            v = x[3:]
            a = u if u is not None else np.zeros(3)
            
            r_new = r + v * dt + 0.5 * a * dt**2
            v_new = v + a * dt
            return np.concatenate([r_new, v_new])
            
        def F_jac(x, dt, u):
            F = np.eye(6)
            F[:3, 3:] = np.eye(3) * dt
            return F
            
        self.ekf.predict(fx, F_jac, dt, u=accel)

    def update_landmark(self, z_obs, landmark_pos):
        """
        Update state using observed relative vector to a known landmark.
        
        Args:
            z_obs (np.ndarray): Measured relative vector [dx, dy, dz] in local frame.
            landmark_pos (np.ndarray): Known position of the landmark in local frame.
        """
        def hx(x):
            # Relative vector: landmark - rover
            return landmark_pos - x[:3]
            
        def H_jac(x):
            H = np.zeros((3, 6))
            H[:, :3] = -np.eye(3)
            return H
            
        self.ekf.update(z_obs, hx, H_jac)

    @property
    def state(self):
        return self.ekf.x
