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
    def propagate(self, r_i: np.ndarray, v_i: np.ndarray, dt: float, **kwargs):
        """
        Propagates the state vector (position and velocity) forward in time.

        Args:
            r_i (np.ndarray): Initial position vector [km] or [m] (consistent units required).
            v_i (np.ndarray): Initial velocity vector [km/s] or [m/s].
            dt (float): Time step to propagate [s].
            **kwargs: Additional arguments specific to the propagator.

        Returns:
            tuple: (r_f, v_f) Final position and velocity vectors.
        """
        pass
