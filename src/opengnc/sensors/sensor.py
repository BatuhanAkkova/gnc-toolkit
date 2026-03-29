"""
Abstract base class for all sensors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np


class Sensor(ABC):
    """
    Abstract Base Class for all sensors.

    Parameters
    ----------
    name : str, optional
        Sensor name. Default "Sensor".
    """

    def __init__(self, name: str = "Sensor") -> None:
        """Initialize sensor base."""
        self.name = name
        self.fault_state: str | None = None
        self.stuck_value: np.ndarray | float | None = None

    @abstractmethod
    def measure(self, *args: Any, **kwargs: Any) -> Any:
        """
        Generate a measurement.

        Parameters
        ----------
        *args : Any
            Input state(s) or signals required by the sensor.
        **kwargs : Any
            Additional parameters.

        Returns
        -------
        Any
            Measured value.
        """
        pass

    def apply_calibration(
        self,
        value: np.ndarray | float,
        misalignment: np.ndarray | None = None,
        scale_factor: np.ndarray | float = 1.0,
        bias: np.ndarray | float | None = None,
    ) -> np.ndarray | float:
        """
        Apply calibration residuals: val_cal = (I + M) * S * val_true + b.

        Parameters
        ----------
        value : np.ndarray | float
            True value to be calibrated.
        misalignment : np.ndarray, optional
            Skew/misalignment matrix (3x3 for vectors).
        scale_factor : np.ndarray | float, optional
            Scale factor error (scalar or vector). Default is 1.0.
        bias : np.ndarray | float, optional
            Constant bias vector or scalar.

        Returns
        -------
        np.ndarray | float
            Calibrated value.
        """
        if isinstance(value, np.ndarray):
            # Misalignment (M)
            if misalignment is not None:
                # Assuming misalignment is a 3x3 matrix (skew)
                value = (np.eye(len(value)) + misalignment) @ value

            # Scale Factor (S)
            value = scale_factor * value

            # Bias (b)
            if bias is not None:
                value = value + bias
        else:
            # Scalar case
            value = scale_factor * value
            if bias is not None:
                value += bias

        return cast(np.ndarray | float, value)

    def apply_fogm_noise(
        self, current_val: np.ndarray | float, sigma: float, tau: float, dt: float
    ) -> np.ndarray | float:
        """
        Apply First-Order Gauss-Markov (FOGM) noise.

        x[k+1] = exp(-dt/tau) * x[k] + sigma * sqrt(1 - exp(-2*dt/tau)) * w[k]

        Parameters
        ----------
        current_val : np.ndarray | float
            Current noise state value.
        sigma : float
            Steady-state standard deviation.
        tau : float
            Correlation time constant (s).
        dt : float
            Time step (s).

        Returns
        -------
        np.ndarray | float
            Updated noise state.
        """
        if sigma == 0 or tau <= 0:
            return cast(np.ndarray | float, current_val)

        phi = np.exp(-dt / tau)
        q = sigma * np.sqrt(1 - np.exp(-2 * dt / tau))

        noise = np.random.normal(0, q, size=np.shape(current_val))
        return cast(np.ndarray | float, phi * current_val + noise)

    def apply_faults(self, value: np.ndarray | float) -> np.ndarray | float:
        """
        Inject faults into the measurement.

        Parameters
        ----------
        value : np.ndarray | float
            Clean measurement value.

        Returns
        -------
        np.ndarray | float
            Faulted measurement value.
        """
        if self.fault_state == "stuck":
            return cast(np.ndarray | float, self.stuck_value if self.stuck_value is not None else value)

        if self.fault_state == "spike":
            spike = np.random.normal(
                0, 100 * np.std(value) if np.std(value) > 0 else 10.0, size=np.shape(value)
            )
            return cast(np.ndarray | float, value + spike)

        if self.fault_state == "noise_increase":
            # Add extra noise
            return cast(
                np.ndarray | float,
                value + np.random.normal(
                0, 10.0 * np.std(value) if np.std(value) > 0 else 1.0, size=np.shape(value)
                ),
            )

        return cast(np.ndarray | float, value)

    def add_gaussian_noise(self, value: np.ndarray | float, std_dev: float) -> np.ndarray | float:
        """
        Helper to add zero-mean Gaussian noise.

        Parameters
        ----------
        value : np.ndarray | float
            Nominal value.
        std_dev : float
            Standard deviation of noise.

        Returns
        -------
        np.ndarray | float
            Noisy value.
        """
        if std_dev is None or std_dev == 0:
            return cast(np.ndarray | float, value)

        noise = np.random.normal(0, std_dev, size=np.shape(value))
        return cast(np.ndarray | float, value + noise)




