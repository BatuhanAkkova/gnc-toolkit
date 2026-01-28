import numpy as np
from gnc_toolkit.sensors.sensor import Sensor
from gnc_toolkit.utils.quat_utils import quat_mult

class StarTracker(Sensor):
    """
    Star Tracker sensor model.
    Measures the attitude quaternion [x, y, z, w].
    """
    def __init__(self, noise_std=0.0, bias=None, name="StarTracker"):
        """
        Args:
            noise_std (float): Standard deviation of noise [rad] (applied as small angle rotation).
            bias (np.ndarray): Constant bias rotation (small angles) [rad]. Shape (3,).
            name (str): Sensor name.
        """
        super().__init__(name)
        self.noise_std = noise_std
        self.bias = bias if bias is not None else np.zeros(3)

    def measure(self, true_quat, **kwargs):
        """
        Simulate star tracker measurement.
        
        Args:
            true_quat (np.ndarray): True attitude quaternion [x, y, z, w].
            
        Returns:
            np.ndarray: Measured quaternion [x, y, z, w].
        """
        # Generate error rotation vector (small angle approximation)
        noise = np.random.normal(0, self.noise_std, 3)
        error_vec = self.bias + noise
        
        # q_err approx [ex/2, ey/2, ez/2, 1] for small angles (x, y, z, w)
        angle = np.linalg.norm(error_vec)
        if angle > 1e-8:
            axis = error_vec / angle
            # [x, y, z, w]
            q_err = np.array([
                axis[0]*np.sin(angle/2), 
                axis[1]*np.sin(angle/2), 
                axis[2]*np.sin(angle/2),
                np.cos(angle/2)
            ])
        else:
            q_err = np.array([0.0, 0.0, 0.0, 1.0])
        
        q_meas = quat_mult(true_quat, q_err)
        
        # Ensure unit quaternion
        q_meas = q_meas / np.linalg.norm(q_meas)
        
        return q_meas
