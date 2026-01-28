from abc import ABC, abstractmethod
import numpy as np

class Sensor(ABC):
    """
    Abstract base class for all sensors.
    """
    def __init__(self, name="Sensor"):
        self.name = name

    @abstractmethod
    def measure(self, true_state, **kwargs):
        """
        Generate a measurement based on the true state.
        
        Args:
            true_state: The true physical state (format depends on sensor type).
            **kwargs: Additional parameters (e.g., time, environment).
            
        Returns:
            Measured value with noise/bias applied.
        """
        pass

    def add_gaussian_noise(self, value, std_dev):
        """
        Helper to add zero-mean Gaussian noise.
        """
        if std_dev is None or std_dev == 0:
            return value
        
        noise = np.random.normal(0, std_dev, size=np.shape(value))
        return value + noise
