"""
Radar / Altimeter sensor model.
"""

import numpy as np
from gnc_toolkit.sensors.sensor import Sensor

class Altimeter(Sensor):
    """
    Radar / Altimeter sensor model.
    Measures height above a reference surface (altitude).
    """
    def __init__(self, noise_std=1.0, bias=0.0, name="Altimeter"):
        """
        Args:
            noise_std (float): Standard deviation of measurement noise [m].
            bias (float): Constant bias in altitude [m].
        """
        super().__init__(name)
        self.noise_std = noise_std
        self.bias = bias

    def measure(self, true_altitude, **kwargs):
        """
        Args:
            true_altitude (float): True altitude above surface [m].
            
        Returns:
            float: Measured altitude [m].
        """
        measured_alt = true_altitude + self.bias + np.random.normal(0, self.noise_std)
        return max(0.0, measured_alt)
