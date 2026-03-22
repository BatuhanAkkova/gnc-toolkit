"""
Feedback Linearization Controller for nonlinear systems.
"""

import numpy as np

class FeedbackLinearization:
    """
    Feedback Linearization Controller.
    
    For a system of the form:
    dot_x = f(x) + g(x) u
    
    Computes the control input u that achieves the desired linear dynamics:
    dot_x = v
    
    The control law is:
    u = g(x)^-1 * (v - f(x))
    """
    def __init__(self, f_func, g_func):
        """
        Initialize the Feedback Linearization Controller.
        
        Args:
            f_func (callable): Function f(x) returning the drift vector.
            g_func (callable): Function g(x) returning the input matrix.
        """
        self.f_func = f_func
        self.g_func = g_func

    def compute_control(self, x, v):
        """
        Compute the linearizing control input.
        
        Args:
            x (np.ndarray): Current state vector.
            v (np.ndarray): Desired linear dynamics input (pseudo-control).
            
        Returns:
            np.ndarray: Control input u.
        """
        x = np.array(x)
        v = np.array(v)
        
        f_val = np.array(self.f_func(x))
        g_val = np.array(self.g_func(x))
        
        # Check dimensions
        if g_val.ndim < 2:
            # Scalar case
            if g_val.size == 1:
                u = (v - f_val) / g_val
            else:
                # Using solve for square matrix stability
                u = np.linalg.solve(g_val, v - f_val)
        else:
             u = np.linalg.solve(g_val, v - f_val)
             
        return u
