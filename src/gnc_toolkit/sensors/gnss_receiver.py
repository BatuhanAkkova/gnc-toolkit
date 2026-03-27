"""
GNSS Receiver sensor model.
"""


import numpy as np

from gnc_toolkit.sensors.sensor import Sensor


class GNSSReceiver(Sensor):
    """
    GNSS Receiver sensor model.

    Simulates position and velocity measurements in ECEF or ECI frame.

    Parameters
    ----------
    pos_noise_std : float, optional
        Position measurement noise standard deviation (m). Default is 10.0.
    vel_noise_std : float, optional
        Velocity measurement noise standard deviation (m/s). Default is 0.1.
    name : str, optional
        Sensor name. Default is "GNSS".
    pos_bias : np.ndarray, optional
        Constant position bias (3,).
    vel_bias : np.ndarray, optional
        Constant velocity bias (3,).
    """

    def __init__(
        self,
        pos_noise_std: float = 10.0,
        vel_noise_std: float = 0.1,
        name: str = "GNSS",
        pos_bias: np.ndarray | None = None,
        vel_bias: np.ndarray | None = None,
        **kwargs
    ) -> None:
        super().__init__(name)
        self.pos_noise_std = pos_noise_std
        self.vel_noise_std = vel_noise_std
        self.pos_bias = np.asarray(pos_bias) if pos_bias is not None else np.zeros(3)
        self.vel_bias = np.asarray(vel_bias) if vel_bias is not None else np.zeros(3)

    def measure(
        self, true_pos: np.ndarray, true_vel: np.ndarray, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate GNSS position and velocity measurements.

        Measurements are simulated by adding Gaussian noise and bias to the true values.
        The coordinate frame (ECEF/ECI) is determined by the input `true_pos` and `true_vel`.

        Parameters
        ----------
        true_pos : np.ndarray
            True position vector (m).
        true_vel : np.ndarray
            True velocity vector (m/s).
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (meas_pos, meas_vel).
        """
        meas_pos = self.add_gaussian_noise(true_pos, self.pos_noise_std) + self.pos_bias
        meas_vel = self.add_gaussian_noise(true_vel, self.vel_noise_std) + self.vel_bias

        # Apply faults if any
        meas_pos = self.apply_faults(meas_pos)
        meas_vel = self.apply_faults(meas_vel)

        return meas_pos, meas_vel
