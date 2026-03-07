import numpy as np
from gnc_toolkit.sensors.sensor import Sensor

class Lidar(Sensor):
    """
    Lidar sensor model.
    Measures range and line-of-sight (LOS) vector to a target point.
    """
    def __init__(self, range_noise_std=0.01, los_noise_std=0.001, name="Lidar"):
        """
        Args:
            range_noise_std (float): Range measurement noise std dev [m].
            los_noise_std (float): Angular noise std dev for LOS vector [rad].
        """
        super().__init__(name)
        self.range_noise_std = range_noise_std
        self.los_noise_std = los_noise_std

    def measure(self, true_relative_pos, **kwargs):
        """
        Args:
            true_relative_pos (np.ndarray): True relative position vector in body frame [m].
            
        Returns:
            tuple: (measured_range, measured_los_vec)
        """
        true_range = np.linalg.norm(true_relative_pos)
        true_los = true_relative_pos / true_range if true_range > 0 else np.zeros(3)
        
        # Add range noise
        measured_range = true_range + np.random.normal(0, self.range_noise_std)
        
        # Add LOS noise (small random rotation)
        if true_range > 0:
            noise_vec = np.random.normal(0, self.los_noise_std, 3)
            measured_los = true_los + noise_vec
            measured_los /= np.linalg.norm(measured_los)
        else:
            measured_los = true_los
            
        return max(0.0, measured_range), measured_los
