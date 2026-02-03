import numpy as np
from typing import Union

class CrossProductLaw:
    """
    Cross-Product Law for magnetic momentum dumping (desaturation).
    
    This controller calculates the required magnetic dipole moment to reduce 
    angular momentum stored in reaction wheels (desaturation), typically while 
    maintaining a specific mission attitude.
    
    The law generates a dipole moment 'm' such that the resulting torque 
    T = m x B opposes the component of the angular momentum error that is 
    perpendicular to the magnetic field.
    
    Control law: m = k * (h_error x B) / |B|^2
    Resulting Torque: T = -k * [h_error - (h_error . b_unit) * b_unit]
    """
    
    def __init__(self, gain: float):
        """
        Initialize the Cross-Product Law controller.
        
        Args:
            gain: The feedback gain k (> 0). [s^-1]
        """
        self.gain = gain
        
    def calculate_control(
        self, 
        h_error: Union[np.ndarray, list], 
        b_field: Union[np.ndarray, list]
    ) -> np.ndarray:
        """
        Calculate the required magnetic dipole moment.
        
        Args:
            h_error: The angular momentum vector to be dumped [Nms].
            b_field: The local magnetic field vector [T].
            
        Returns:
            Magnetic dipole moment vector (m) [Am^2]. 
        """
        h = np.array(h_error, dtype=float)
        b = np.array(b_field, dtype=float)
        
        b_sq = np.dot(b, b)
        
        # Handle zero field or extremely weak fields to avoid division by zero
        if b_sq < 1e-18:
            return np.zeros(3)
            
        # m = k * (h x B) / |B|^2
        dipole_moment = (self.gain / b_sq) * np.cross(h, b)
        
        return dipole_moment
