"""
Proportional rate damping controller for torque-based detumbling.
"""


import numpy as np


class RateDampingControl:
    r"""
    Proportional angular rate damping controller for spacecraft detumbling.

    Generates torque commands to reduce the spacecraft's angular rates,
    typically using thrusters or other active actuators.

    Control law: $T = -K \omega$

    Parameters
    ----------
    gain : float
        Proportional damping gain $K$ ($K > 0$).
    max_torque : float, optional
        Maximum torque command magnitude allowed (N-m). Default is None.
    """

    def __init__(self, gain: float, max_torque: float | None = None) -> None:
        """Initialize the rate damping controller."""
        self.gain = gain
        self.max_torque = max_torque

    def compute_torque(self, omega: np.ndarray) -> np.ndarray:
        """
        Compute the commanded damping torque.

        Parameters
        ----------
        omega : np.ndarray
            Measured angular velocity vector in the body frame (3,).
            Units: [rad/s].

        Returns
        -------
        np.ndarray
            Commanded torque vector in the body frame (3,). Units: [N-m].
        """
        # Linear damping: T = -k * omega
        torque_raw = -self.gain * np.asarray(omega)

        # Apply torque saturation
        if self.max_torque is not None:
            norm_t = np.linalg.norm(torque_raw)
            if norm_t > self.max_torque:
                return torque_raw * (self.max_torque / norm_t)

        return torque_raw




