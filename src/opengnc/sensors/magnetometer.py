"""
Magnetometer sensor model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from opengnc.sensors.sensor import Sensor


class Magnetometer(Sensor):
    """
    Magnetometer sensor model.

    Measures magnetic field vector in body frame.

    Parameters
    ----------
    noise_std : float, optional
        Standard deviation of measurement noise (Tesla). Default is 0.0.
    bias : np.ndarray, optional
        Hard-iron bias vector (Tesla). Default is zero vector.
    misalignment : np.ndarray, optional
        3x3 soft-iron / misalignment matrix. Default is None.
    scale_factor : float | np.ndarray, optional
        Scale factor error. Default is 1.0.
    name : str, optional
        Sensor name. Default is "Magnetometer".
    """

    def __init__(
        self,
        noise_std: float = 0.0,
        bias: np.ndarray | None = None,
        misalignment: np.ndarray | None = None,
        scale_factor: float | np.ndarray = 1.0,
        name: str = "Magnetometer",
    ) -> None:
        super().__init__(name)
        self.noise_std = noise_std
        self.bias = bias if bias is not None else np.zeros(3)
        self.misalignment = misalignment
        self.scale_factor = scale_factor

    def measure(
        self, true_mag_vec_body: np.ndarray | None = None, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Generate magnetic field measurement.

        Parameters
        ----------
        true_mag_vec_body : np.ndarray
            True magnetic field vector in body frame (Tesla).
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        np.ndarray
            Measured magnetic field vector (Tesla).
        """
        # Apply calibration (soft iron, misalignment, hard iron bias)
        if true_mag_vec_body is None:
            if not args:
                raise ValueError("true_mag_vec_body is required.")
            true_mag_vec_body = np.asarray(args[0])
        calibrated = self.apply_calibration(
            true_mag_vec_body, self.misalignment, self.scale_factor, self.bias
        )

        # Add noise
        measured = self.add_gaussian_noise(calibrated, self.noise_std)

        # Apply faults
        measured = np.asarray(self.apply_faults(measured), dtype=float)

        return measured




