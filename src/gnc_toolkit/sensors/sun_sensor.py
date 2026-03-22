"""
Sun Sensor model.
"""

import numpy as np
from gnc_toolkit.sensors.sensor import Sensor

class SunSensor(Sensor):
    """
    Sun Sensor model.
    Measures the sun vector in the body frame.
    """
    def __init__(self, noise_std=0.0, bias=None, misalignment=None, scale_factor=1.0, name="SunSensor"):
        """
        Args:
            noise_std (float): Standard deviation of noise [rad] or unitless depending on vector norm.
            bias (np.ndarray): Bias vector to add.
            misalignment (np.ndarray): 3x3 misalignment matrix.
            scale_factor (float or np.ndarray): Scale factor error.
        """
        super().__init__(name)
        self.noise_std = noise_std
        self.bias = bias if bias is not None else np.zeros(3)
        self.misalignment = misalignment
        self.scale_factor = scale_factor

    def measure(self, true_sun_vec_body, **kwargs):
        """
        Args:
            true_sun_vec_body (np.ndarray): True sun vector in body frame.
        """
        # Apply calibration
        calibrated = self.apply_calibration(true_sun_vec_body, self.misalignment, self.scale_factor, self.bias)
        
        # Add noise
        measured_vec = self.add_gaussian_noise(calibrated, self.noise_std)
        
        # Apply faults
        measured_vec = self.apply_faults(measured_vec)
        
        # Normalize
        norm = np.linalg.norm(measured_vec)
        if norm > 0:
            measured_vec = measured_vec / norm
            
        return measured_vec
