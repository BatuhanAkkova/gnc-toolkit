from abc import ABC, abstractmethod
import numpy as np

class Actuator(ABC):
    """
    Abstract base class for all actuators.
    """
    def __init__(self, name="Actuator", saturation=None, deadband=None):
        """
        Args:
            name (str): Actuator name.
            saturation (float or tuple): Max output magnitude or (min, max).
            deadband (float): Input values below this magnitude result in zero output.
        """
        self.name = name
        self.saturation = saturation
        self.deadband = deadband

    @abstractmethod
    def command(self, signal, **kwargs):
        """
        Calculate the actuator output based on the command signal.
        
        Args:
            signal: The commanded input (e.g., torque, dipole, voltage).
            kwargs: Additional state info (e.g., current speed, environment).
            
        Returns:
            The actual output applied to the system.
        """
        pass

    def apply_saturation(self, value):
        """
        Apply saturation limits to value.
        """
        if self.saturation is None:
            return value
        
        if isinstance(self.saturation, (int, float)):
            limit = abs(self.saturation)
            return np.clip(value, -limit, limit)
        elif isinstance(self.saturation, (list, tuple)) and len(self.saturation) == 2:
            return np.clip(value, self.saturation[0], self.saturation[1])
        return value

    def apply_deadband(self, value):
        """
        Apply deadband to value.
        """
        if self.deadband is None or self.deadband == 0:
            return value
        
        if abs(value) < self.deadband:
            return 0.0
        return value
