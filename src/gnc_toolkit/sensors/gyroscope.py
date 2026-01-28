import numpy as np
from gnc_toolkit.sensors.sensor import Sensor

class Gyroscope(Sensor):
    """
    Gyroscope sensor model.
    Measures angular velocity: w_meas = w_true + bias + noise
    Includes bias instability (random walk).
    """
    def __init__(self, noise_std=0.0, bias_stability=0.0, initial_bias=None, dt=0.1, name="Gyroscope"):
        """
        Args:
            noise_std (float): Angle Random Walk (ARW) coefficient or white noise std dev [rad/s].
            bias_stability (float): Bias instability / Rate Random Walk std dev [rad/s/sqrt(s)?].
                                    Simplified: std dev of the random walk step per integration step.
            initial_bias (np.ndarray): Initial constant bias [rad/s].
            dt (float): Time step for bias propagation [s].
        """
        super().__init__(name)
        self.noise_std = noise_std
        self.bias_stability = bias_stability
        self.current_bias = initial_bias if initial_bias is not None else np.zeros(3)
        self.dt = dt

    def measure(self, true_omega, **kwargs):
        """
        Args:
            true_omega (np.ndarray): True angular velocity [rad/s].
            kwargs: Can contain 'dt' to update time step if variable.
        """
        dt = kwargs.get('dt', self.dt)
        
        # Propagate bias (Random Walk)
        # bias[k+1] = bias[k] + noise_b
        if self.bias_stability > 0:
            walk_std = self.bias_stability * np.sqrt(dt) 
            self.current_bias += np.random.normal(0, walk_std, 3)
            
        # Add noise (measurement noise / ARW)
        measurement_noise = np.random.normal(0, self.noise_std, 3)
        
        measured_omega = true_omega + self.current_bias + measurement_noise
        
        return measured_omega
