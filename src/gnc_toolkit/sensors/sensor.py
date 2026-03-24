"""
Abstract base class for all sensors.
"""

from abc import ABC, abstractmethod

import numpy as np


class Sensor(ABC):
    """
    Abstract base class for all sensors.
    """

    def __init__(self, name="Sensor"):
        self.name = name
        self.fault_state = None  # None, "stuck", "spike", "noise_increase"
        self.stuck_value = None

    @abstractmethod
    def measure(self, true_state, **kwargs):
        """
        Generate a measurement based on the true state.

        Args:
            true_state: The true physical state (format depends on sensor type).
            **kwargs: Additional parameters (e.g., time, environment).

        Returns
        -------
            Measured value with noise/bias applied.
        """
        pass

    def apply_calibration(self, value, misalignment=None, scale_factor=1.0, bias=None):
        """
        Applies calibration residuals:
        val_cal = (I + M) * S * val_true + b
        """
        if isinstance(value, np.ndarray):
            # Misalignment (M)
            if misalignment is not None:
                # Assuming misalignment is a 3x3 matrix (skew)
                value = (np.eye(len(value)) + misalignment) @ value

            # Scale Factor (S)
            if isinstance(scale_factor, np.ndarray):
                value = scale_factor * value
            else:
                value = scale_factor * value

            # Bias (b)
            if bias is not None:
                value = value + bias
        else:
            # Scalar case
            value = scale_factor * value
            if bias is not None:
                value += bias

        return value

    def apply_fogm_noise(self, current_val, sigma, tau, dt):
        """
        Applies First-Order Gauss-Markov (FOGM) noise.
        x[k+1] = exp(-dt/tau) * x[k] + sigma * sqrt(1 - exp(-2*dt/tau)) * w[k]
        """
        if sigma == 0 or tau <= 0:
            return current_val

        phi = np.exp(-dt / tau)
        q = sigma * np.sqrt(1 - np.exp(-2 * dt / tau))

        noise = np.random.normal(0, q, size=np.shape(current_val))
        return phi * current_val + noise

    def apply_faults(self, value):
        """
        Injects faults into the measurement.
        """
        if self.fault_state == "stuck":
            return self.stuck_value if self.stuck_value is not None else value

        if self.fault_state == "spike":
            spike = np.random.normal(
                0, 100 * np.std(value) if np.std(value) > 0 else 10.0, size=np.shape(value)
            )
            return value + spike

        if self.fault_state == "noise_increase":
            # Add extra noise
            return value + np.random.normal(
                0, 10.0 * np.std(value) if np.std(value) > 0 else 1.0, size=np.shape(value)
            )

        return value

    def add_gaussian_noise(self, value, std_dev):
        """
        Helper to add zero-mean Gaussian noise.
        """
        if std_dev is None or std_dev == 0:
            return value

        noise = np.random.normal(0, std_dev, size=np.shape(value))
        return value + noise
