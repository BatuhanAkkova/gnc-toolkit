"""
Earth / Horizon sensor model.
"""

import numpy as np

from gnc_toolkit.sensors.sensor import Sensor


class HorizonSensor(Sensor):
    """
    Earth / Horizon sensor model.

    Measures the nadir vector in the body frame.

    Parameters
    ----------
    noise_std : float, optional
        Standard deviation of measurement noise (rad). Default is 0.01.
    bias : np.ndarray, optional
        Constant bias in roll/pitch equivalent (rad). Default is [0, 0].
    name : str, optional
        Sensor name. Default is "HorizonSensor".
    """

    def __init__(
        self,
        noise_std: float = 0.01,
        bias: np.ndarray | None = None,
        name: str = "HorizonSensor",
    ):
        super().__init__(name)
        self.noise_std = noise_std
        self.bias = bias if bias is not None else np.zeros(2)  # [roll_error, pitch_error]

    def measure(self, true_nadir_vec: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate nadir vector measurement.

        Parameters
        ----------
        true_nadir_vec : np.ndarray
            True nadir unit vector in body frame.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        np.ndarray
            Measured nadir unit vector in body frame.
        """
        n = true_nadir_vec / np.linalg.norm(true_nadir_vec)

        noise_vec = np.random.normal(0, self.noise_std, 3)
        meas_n = n + noise_vec

        if np.linalg.norm(self.bias) > 0:
            # Bias applied as component offsets (roll/pitch approximation)
            meas_n[0] += self.bias[0]
            meas_n[1] += self.bias[1]

        return meas_n / np.linalg.norm(meas_n)
