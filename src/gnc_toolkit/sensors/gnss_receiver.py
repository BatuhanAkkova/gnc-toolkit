"""
GNSS Receiver sensor model.
"""

import numpy as np

from gnc_toolkit.sensors.sensor import Sensor


class GNSSReceiver(Sensor):
    """
    GNSS Receiver sensor model.
    Measures position and velocity:
    r_meas = r_true + pos_bias + pos_noise
    v_meas = v_true + vel_bias + vel_noise
    """

    def __init__(
        self, pos_noise_std=1.0, vel_noise_std=0.01, pos_bias=None, vel_bias=None, name="GNSS"
    ):
        """
        Args:
            pos_noise_std (float): Standard deviation of position noise [m].
            vel_noise_std (float): Standard deviation of velocity noise [m/s].
            pos_bias (np.ndarray): Constant position bias [m].
            vel_bias (np.ndarray): Constant velocity bias [m/s].
        """
        super().__init__(name)
        self.pos_noise_std = pos_noise_std
        self.vel_noise_std = vel_noise_std
        self.pos_bias = pos_bias if pos_bias is not None else np.zeros(3)
        self.vel_bias = vel_bias if vel_bias is not None else np.zeros(3)

    def measure(self, true_r, true_v, **kwargs):
        """
        Args:
            true_r (np.ndarray): True position vector [m].
            true_v (np.ndarray): True velocity vector [m/s].

        Returns
        -------
            tuple: (measured_r, measured_v)
        """
        meas_r = true_r + self.pos_bias + np.random.normal(0, self.pos_noise_std, 3)
        meas_v = true_v + self.vel_bias + np.random.normal(0, self.vel_noise_std, 3)

        return meas_r, meas_v
