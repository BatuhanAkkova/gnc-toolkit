"""
Magnetometer sensor model.
"""

import numpy as np

from gnc_toolkit.sensors.sensor import Sensor


class Magnetometer(Sensor):
    """
    Magnetometer sensor model.
    Measures magnetic field vector in body frame.
    """

    def __init__(
        self, noise_std=0.0, bias=None, misalignment=None, scale_factor=1.0, name="Magnetometer"
    ):
        """
        Args:
            noise_std (float): Standard deviation of measurement noise [Tesla].
            bias (np.ndarray): hard-iron bias [Tesla].
            misalignment (np.ndarray): 3x3 soft-iron / misalignment matrix.
            scale_factor (float or np.ndarray): Scale factor error.
        """
        super().__init__(name)
        self.noise_std = noise_std
        self.bias = bias if bias is not None else np.zeros(3)
        self.misalignment = misalignment
        self.scale_factor = scale_factor

    def measure(self, true_mag_vec_body, **kwargs):
        """
        Args:
            true_mag_vec_body (np.ndarray): True magnetic field vector in body frame [Tesla].
        """
        # Apply calibration (soft iron, misalignment, hard iron bias)
        calibrated = self.apply_calibration(
            true_mag_vec_body, self.misalignment, self.scale_factor, self.bias
        )

        # Add noise
        measured = self.add_gaussian_noise(calibrated, self.noise_std)

        # Apply faults
        measured = self.apply_faults(measured)

        return measured
