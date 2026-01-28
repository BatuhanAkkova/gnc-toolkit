import numpy as np
from gnc_toolkit.sensors.sensor import Sensor

class SunSensor(Sensor):
    """
    Sun Sensor model.
    Measures the sun vector in the body frame.
    """
    def __init__(self, noise_std=0.0, bias=None, name="SunSensor"):
        """
        Args:
            noise_std (float): Standard deviation of noise [rad] or unitless depending on vector norm.
                               Here assuming angular noise applied to vector.
            bias (np.ndarray): Bias vector to add.
        """
        super().__init__(name)
        self.noise_std = noise_std
        self.bias = bias if bias is not None else np.zeros(3)

    def measure(self, true_sun_vec_body, **kwargs):
        """
        Args:
            true_sun_vec_body (np.ndarray): True sun vector in body frame.
        """
        # Basic additive noise model for vector measurement
        measured_vec = true_sun_vec_body + self.bias
        measured_vec = self.add_gaussian_noise(measured_vec, self.noise_std)
        
        # Normalize
        norm = np.linalg.norm(measured_vec)
        if norm > 0:
            measured_vec = measured_vec / norm
            
        return measured_vec
