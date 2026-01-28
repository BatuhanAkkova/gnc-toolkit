import numpy as np
from typing import Union

class BDot:
    """
    B-Dot controller for magnetic detumbling.
    
    The control law calculates a magnetic dipole moment 'm' to dampen the angular velocity
    of the spacecraft.
    
    Control law: m = -K * B_dot
    where B_dot is the time derivative of the magnetic field vector.
    """
    
    def __init__(self, gain: float):
        """
        Initialize the B-Dot controller.
        
        Args:
            gain: The feedback gain K (> 0). [A*m^2 * s / T]
        """
        self.gain = gain
        
    def calculate_control(self, b_dot: Union[np.ndarray, list]) -> np.ndarray:
        """
        Calculate the required magnetic dipole moment based on the rate of change of B-field.
        
        Args:
            b_dot: The time derivative of the magnetic field vector (dB/dt) in Body frame. Shape (3,)
                   
        Returns:
            Magnetic dipole moment vector (m) in Body frame. Shape (3,)
        """
        b_dot_vec = np.array(b_dot, dtype=float)
        
        # Standard law: m = -K * B_dot
        dipole_moment = -self.gain * b_dot_vec
        
        return dipole_moment
    
    def calculate_control_discrete(
        self, 
        b_field_current: Union[np.ndarray, list], 
        b_field_prev: Union[np.ndarray, list], 
        dt: float
    ) -> np.ndarray:
        """
        Calculate control using discrete finite difference of B-field.
        
        Args:
            b_field_current: Current magnetic field vector measurement (Body frame)
            b_field_prev: Previous magnetic field vector measurement (Body frame)
            dt: Time step between measurements (seconds)
            
        Returns:
            Magnetic dipole moment vector (m) in Body frame.
        """
        b_curr = np.array(b_field_current, dtype=float)
        b_prev = np.array(b_field_prev, dtype=float)
        
        if dt <= 0:
            return np.zeros(3)
            
        b_dot_est = (b_curr - b_prev) / dt
        return self.calculate_control(b_dot_est)
