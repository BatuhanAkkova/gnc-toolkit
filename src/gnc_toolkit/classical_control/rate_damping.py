"""
Proportional rate damping controller for torque-based detumbling.
"""

import numpy as np


class RateDampingControl:
    """
    Simple proportional rate damping control.

    This controller generates torque commands to reduce the angular
    velocity of the spacecraft.

    The control law is:
    T = -k * omega
    where:
        T is the commanded torque
        k is the proportional gain (scalar or 3x3 diagonal matrix)
        omega is the measured angular velocity
    """

    def __init__(self, gain: float, max_torque: float | None = None):
        """
        Initialize the Rate Damping controller.

        Args:
            gain: Proportional gain (damping coefficient)
            max_torque: Maximum torque command (saturation per axis or total norm)
        """
        self.gain = gain
        self.max_torque = max_torque

    def compute_torque(self, omega: np.ndarray) -> np.ndarray:
        """
        Compute the damping torque.

        Args:
            omega: Angular velocity in body frame [rad/s] (3,)

        Returns
        -------
            Commanded torque [Nm] (3,)
        """
        omega = np.asarray(omega)

        # T = -k * omega
        torque_raw = -self.gain * omega

        if self.max_torque is not None:
            norm_torque = np.linalg.norm(torque_raw)
            if norm_torque > self.max_torque:
                return torque_raw * (self.max_torque / norm_torque)

        return torque_raw
