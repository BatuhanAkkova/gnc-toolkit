"""
Abstract base class for orbit propagators.
"""

from abc import ABC, abstractmethod

import numpy as np


class Propagator(ABC):
    """
    Abstract base class for orbit propagators.
    """

    @abstractmethod
    def propagate(
        self, r_i: np.ndarray, v_i: np.ndarray, dt: float, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Propagates the state vector (position and velocity) forward in time.

        Parameters
        ----------
        r_i : np.ndarray
            Initial position vector (m).
        v_i : np.ndarray
            Initial velocity vector (m/s).
        dt : float
            Time duration for propagation (s).
        **kwargs : dict
            Additional arguments specific to the propagator implementation.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (r_f, v_f).
            r_f : Final position vector (m).
            v_f : Final velocity vector (m/s).
        """
        pass
