import numpy as np
from gnc_toolkit.sensors.sensor import Sensor

class Magnetometer(Sensor):
    """
    Magnetometer sensor model.
    Measures magnetic field vector in body frame.
    """
    def __init__(self, noise_std=0.0, bias=None, scale_factor=None, name="Magnetometer"):
        super().__init__(name)
        self.noise_std = noise_std
        self.bias = bias if bias is not None else np.zeros(3)
        self.scale_factor = scale_factor if scale_factor is not None else np.eye(3)

    def measure(self, true_mag_vec_body, **kwargs):
        """
        Args:
            true_mag_vec_body (np.ndarray): True magnetic field vector in body frame [Tesla].
        """
        # Apply scale factor (soft iron / misalignment)
        scaled = self.scale_factor @ true_mag_vec_body
        
        # Add bias (hard iron)
        biased = scaled + self.bias
        
        # Add noise
        measured = self.add_gaussian_noise(biased, self.noise_std)
        
        return measured
