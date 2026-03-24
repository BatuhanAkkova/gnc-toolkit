"""
Coarse Sun Sensor Array model.
"""

import numpy as np
from gnc_toolkit.sensors.sensor import Sensor

class CoarseSunSensorArray(Sensor):
    r"""
    Array of Coarse Sun Sensors (CSS).
    Each CSS measures the cosine of the angle between the sun vector and its boresight.
    $I = I_{max} \cos(\theta) = I_{max} (S \cdot N)$
    """
    def __init__(self, boresights=None, i_max=1.0, noise_std=0.01, name="CSSArray"):
        """
        Args:
            boresights (list of np.ndarray): Unit vectors representing the boresight 
                                            direction of each CSS in the body frame.
            i_max (float): Maximum current/output when sun is aligned with boresight.
            noise_std (float): Standard deviation of measurement noise.
        """
        super().__init__(name)
        if boresights is None:
            # Default to 6 faces of a cube
            boresights = [
                np.array([1, 0, 0]), np.array([-1, 0, 0]),
                np.array([0, 1, 0]), np.array([0, -1, 0]),
                np.array([0, 0, 1]), np.array([0, 0, -1])
            ]
        self.boresights = [b / np.linalg.norm(b) for b in boresights]
        self.i_max = i_max
        self.noise_std = noise_std

    def measure(self, true_sun_vec, **kwargs):
        """
        Args:
            true_sun_vec (np.ndarray): True sun unit vector in body frame.
            
        Returns:
            np.ndarray: Array of measurements from each CSS.
        """
        s = true_sun_vec / np.linalg.norm(true_sun_vec)
        measurements = []
        for n in self.boresights:
            cos_theta = np.dot(s, n)
            # CSS only works if sun is in front of the sensor
            i_meas = self.i_max * max(0.0, cos_theta)
            i_meas += np.random.normal(0, self.noise_std)
            measurements.append(max(0.0, i_meas))
            
        return np.array(measurements)
