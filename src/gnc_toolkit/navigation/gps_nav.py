"""
Navigation using GNSS (GPS) position and velocity measurements.
"""

import numpy as np

from .orbit_determination import OrbitDeterminationEKF


class GPSNavigation(OrbitDeterminationEKF):
    """
    Navigation using GNSS (GPS) measurements.
    Processes position and velocity updates.
    """

    def update_gps(self, r_meas, v_meas=None, R_gps=None):
        """
        Update state using GPS measurements.

        Args:
            r_meas (np.ndarray): Measured position [x, y, z] in ECI [m].
            v_meas (np.ndarray, optional): Measured velocity [vx, vy, vz] in ECI [m/s].
            R_gps (np.ndarray, optional): Custom measurement noise covariance.
        """
        if v_meas is not None:
            # Update with both position and velocity (6D measurement)
            z = np.concatenate([r_meas, v_meas])

            def hx(x):
                return x  # Measurement is the full state (pos, vel)

            def H_jac(x):
                return np.eye(6)

            if R_gps is None:
                # Default GPS R: 3x3 for pos, 3x3 for vel
                R = np.eye(6)
                R[:3, :3] = self.ekf.R
                R[3:, 3:] = self.ekf.R * 0.01
            else:
                R = R_gps

            self.ekf.update(z, hx, H_jac, R=R)
        else:
            # Fallback to position-only update (defined in base class)
            self.update(r_meas)
