"""
Star Tracker sensor model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from opengnc.sensors.sensor import Sensor
from opengnc.utils.quat_utils import quat_mult


class StarTracker(Sensor):
    """
    Star Tracker Attitude Sensor.

    Measures the attitude quaternion [x, y, z, w].

    Parameters
    ----------
    noise_std : float, optional
        Standard deviation of noise (rad). Default 0.0.
    bias : np.ndarray | None, optional
        Constant bias rotation vector (rad).
    name : str, optional
        Sensor name. Default "StarTracker".
    """

    def __init__(self, noise_std: float = 0.0, bias: np.ndarray | None = None, name: str = "StarTracker") -> None:
        """Initialize star tracker."""
        super().__init__(name)
        self.noise_std = noise_std
        self.bias = np.asarray(bias) if bias is not None else np.zeros(3)

    def measure(self, true_quat: np.ndarray | None = None, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Simulate attitude measurement with error quaternion.

        Parameters
        ----------
        true_quat : np.ndarray
            True attitude quaternion [x, y, z, w].
        **kwargs : Any
            Additional parameters.

        Returns
        -------
        np.ndarray
            Measured quaternion [x, y, z, w].
        """
        if true_quat is None:
            if not args:
                raise ValueError("true_quat is required.")
            true_quat = np.asarray(args[0])
        tq = np.asarray(true_quat)
        noise = np.random.normal(0, self.noise_std, 3)
        error_vec = self.bias + noise

        angle = np.linalg.norm(error_vec)
        if angle > 1e-8:
            axis = error_vec / angle
            q_err = np.array([
                axis[0] * np.sin(angle / 2),
                axis[1] * np.sin(angle / 2),
                axis[2] * np.sin(angle / 2),
                np.cos(angle / 2)
            ])
        else:
            q_err = np.array([0.0, 0.0, 0.0, 1.0])

        q_meas = quat_mult(tq, q_err)
        q_meas = q_meas / np.linalg.norm(q_meas)

        # Apply faults from base class
        q_meas = np.asarray(self.apply_faults(q_meas), dtype=float)
        return np.asarray(q_meas / np.linalg.norm(q_meas), dtype=float)




