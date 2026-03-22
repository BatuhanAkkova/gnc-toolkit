"""
Sliding Mode Controller (SMC) implementation.
"""

import numpy as np

class SlidingModeController:
    """
    Sliding Mode Controller (SMC).
    
    Implements a control law of the form:
    u = u_eq(x, t) - K * sign(s(x, t))
    """
    def __init__(self, surface_func, k_gain, equivalent_control_func=None, chattering_reduction=True, boundary_layer=0.1):
        """
        Initialize the Sliding Mode Controller.
        
        Args:
            surface_func (callable): Function s(x, t) returning the scalar sliding surface value.
            k_gain (float): Switching gain K.
            equivalent_control_func (callable, optional): Function u_eq(x, t) returning the equivalent control. 
                                                          Defaults to returning 0.
            chattering_reduction (bool): If True, use a saturation function instead of sign().
            boundary_layer (float): The thickness 'phi' of the boundary layer for saturation.
        """
        self.surface_func = surface_func
        self.k_gain = k_gain
        self.eq_func = equivalent_control_func if equivalent_control_func else lambda x, t: 0.0
        self.use_sat = chattering_reduction
        self.phi = boundary_layer

    def compute_control(self, x, t=0):
        """
        Compute the control input.
        
        Args:
            x (np.ndarray): State vector.
            t (float): Time.
            
        Returns:
            float or np.ndarray: Control input u.
        """
        s = self.surface_func(x, t)
        
        if self.use_sat:
            # Saturation function: sat(s/phi)
            val = s / self.phi
            # Clip handles scalars and arrays
            switching_term = np.clip(val, -1.0, 1.0)
        else:
            switching_term = np.sign(s)
            
        u_eq = self.eq_func(x, t)
        u = u_eq - self.k_gain * switching_term
        
        return u
