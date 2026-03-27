"""
Inertial Measurement Unit (IMU) with Accelerometer and Gyroscope.
"""

from typing import Any

import numpy as np

from opengnc.sensors.gyroscope import Gyroscope
from opengnc.sensors.sensor import Sensor


class Accelerometer(Sensor):
    """
    Accelerometer sensor model.

    Measures non-gravitational acceleration: a_meas = a_true + bias + noise.

    Parameters
    ----------
    noise_std : float, optional
        Standard deviation of measurement noise (m/s^2). Default is 0.0.
    bias : np.ndarray, optional
        Constant bias vector (m/s^2). Default is zero vector.
    scale_factor : float, optional
        Scale factor error (1.0 = no error). Default is 1.0.
    name : str, optional
        Sensor name. Default is "Accelerometer".
    """

    def __init__(
        self,
        noise_std: float = 0.0,
        bias: np.ndarray | None = None,
        scale_factor: float = 1.0,
        name: str = "Accelerometer",
    ) -> None:
        """
        Initialize accelerometer parameters.

        Parameters
        ----------
        noise_std : float, optional
            Standard deviation of Gaussian noise ($m/s^2$). Default 0.0.
        bias : np.ndarray | None, optional
            Constant bias vector ($m/s^2$).
        scale_factor : float, optional
            Scale factor error. Default 1.0.
        name : str, optional
            Sensor name. Default "Accelerometer".
        """
        super().__init__(name)
        self.noise_std = noise_std
        self.bias = np.asarray(bias) if bias is not None else np.zeros(3)
        self.scale_factor = scale_factor

    def measure(self, true_accel: np.ndarray, **kwargs: Any) -> np.ndarray:
        r"""
        Generate acceleration measurement.

        Equation:
        $\mathbf{a}_{meas} = S \mathbf{a}_{true} + \mathbf{b} + \mathbf{\eta}$

        Parameters
        ----------
        true_accel : np.ndarray
            True non-gravitational acceleration ($m/s^2$).
        **kwargs : Any
            Additional parameters.

        Returns
        -------
        np.ndarray
            Measured acceleration vector ($m/s^2$).
        """
        noise = np.random.normal(0, self.noise_std, 3)
        measured_accel = self.scale_factor * np.asarray(true_accel) + self.bias + noise
        return measured_accel


class IMU(Sensor):
    """
    Inertial Measurement Unit (IMU) combining Gyroscope and Accelerometer.

    Parameters
    ----------
    gyro_params : dict, optional
        Parameters for Gyroscope initialization.
    accel_params : dict, optional
        Parameters for Accelerometer initialization.
    name : str, optional
        Sensor name. Default is "IMU".
    """

    def __init__(
        self,
        gyro_params: dict | None = None,
        accel_params: dict | None = None,
        name: str = "IMU",
    ) -> None:
        super().__init__(name)
        gyro_p = gyro_params if gyro_params is not None else {}
        accel_p = accel_params if accel_params is not None else {}

        self.gyro = Gyroscope(**gyro_p)
        self.accel = Accelerometer(**accel_p)

    def measure(
        self, true_omega: np.ndarray, true_accel: np.ndarray, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate dual IMU measurements.

        Parameters
        ----------
        true_omega : np.ndarray
            True angular velocity (rad/s).
        true_accel : np.ndarray
            True non-gravitational acceleration ($m/s^2$).
        **kwargs : Any
            Additional parameters.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (measured_omega, measured_accel).
        """
        meas_omega = self.gyro.measure(true_omega, **kwargs)
        meas_accel = self.accel.measure(true_accel, **kwargs)
        return meas_omega, meas_accel




