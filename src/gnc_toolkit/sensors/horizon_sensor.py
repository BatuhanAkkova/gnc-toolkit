import numpy as np
from gnc_toolkit.sensors.sensor import Sensor

class HorizonSensor(Sensor):
    """
    Earth / Horizon sensor model.
    Measures the nadir vector in the body frame.
    """
    def __init__(self, noise_std=0.01, bias=None, name="HorizonSensor"):
        """
        Args:
            noise_std (float): Standard deviation of measurement noise [rad].
            bias (np.ndarray): Constant bias in roll/pitch equivalent [rad].
        """
        super().__init__(name)
        self.noise_std = noise_std
        self.bias = bias if bias is not None else np.zeros(2) # [roll_error, pitch_error]

    def measure(self, true_nadir_vec, **kwargs):
        """
        Args:
            true_nadir_vec (np.ndarray): True nadir unit vector in body frame.
            
        Returns:
            np.ndarray: Measured nadir unit vector in body frame.
        """
        # Simplified model: Add noise/bias directly to the vector components
        # or treat as small rotations.
        n = true_nadir_vec / np.linalg.norm(true_nadir_vec)
        
        # Add noise to lateral components (assuming boresight is roughly Z)
        # For a general nadir vector, we can use a random rotation.
        noise_vec = np.random.normal(0, self.noise_std, 3)
        meas_n = n + noise_vec
        
        # Add bias (simple offset)
        if np.linalg.norm(self.bias) > 0:
            # This is a very rough bias model for a vector
            meas_n[0] += self.bias[0]
            meas_n[1] += self.bias[1]
            
        return meas_n / np.linalg.norm(meas_n)
