"""
Sun Sensor model.
"""

import numpy as np

from gnc_toolkit.sensors.sensor import Sensor


class SunSensor(Sensor):
    """
    Sun Sensor model.

    Measures the sun vector in the body frame.

    Parameters
    ----------
    noise_std : float, optional
        Standard deviation of noise (rad or unitless depending on vector norm). Default is 0.0.
    bias : np.ndarray, optional
        Bias vector to add. Default is zero vector.
    misalignment : np.ndarray, optional
        3x3 misalignment matrix. Default is None.
    scale_factor : float | np.ndarray, optional
        Scale factor error. Default is 1.0.
    name : str, optional
        Sensor name. Default is "SunSensor".
    """

    def __init__(
        self,
        noise_std: float = 0.0,
        bias: np.ndarray | None = None,
        misalignment: np.ndarray | None = None,
        scale_factor: float | np.ndarray = 1.0,
        name: str = "SunSensor",
    ) -> None:
        super().__init__(name)
        self.noise_std = noise_std
        self.bias = bias if bias is not None else np.zeros(3)
        self.misalignment = misalignment
        self.scale_factor = scale_factor

    def measure(self, true_sun_vec_body: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate sun vector measurement.

        Parameters
        ----------
        true_sun_vec_body : np.ndarray
            True sun vector in body frame.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        np.ndarray
            Measured (and normalized) sun vector in body frame.
        """
        # Apply calibration
        calibrated = self.apply_calibration(
            true_sun_vec_body, self.misalignment, self.scale_factor, self.bias
        )

        # Add noise
        measured_vec = self.add_gaussian_noise(calibrated, self.noise_std)

        # Apply faults
        measured_vec = self.apply_faults(measured_vec)

        # Normalize
        norm = np.linalg.norm(measured_vec)
        if norm > 0:
            measured_vec = measured_vec / norm

        return measured_vec
