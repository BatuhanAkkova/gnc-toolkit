"""
Backstepping Controller for generic 2nd order nonlinear systems.
"""

import numpy as np

class BacksteppingController:
    """
    Backstepping Controller for generic 2nd order nonlinear systems.
    
    System model:
    x1_dot = x2
    x2_dot = f(x) + g(x) * u
    
    where x = [x1, x2].
    Commonly used for mechanical systems with rigid body dynamics.
    """
    def __init__(self, f_func, g_func, k1, k2):
        """
        Initialize the Backstepping Controller.
        
        Args:
            f_func (callable): Function returns f(x1, x2). scalar/array [n].
            g_func (callable): Function returns g(x1, x2). matrix/scalar [n x m].
            k1 (np.ndarray or float): Gain matrix for tracking error e1.
            k2 (np.ndarray or float): Gain matrix for virtual error e2.
        """
        self.f = f_func
        self.g = g_func
        self.k1 = k1
        self.k2 = k2

    def compute_control(self, x1, x2, x1_d, x1_dot_d, x1_ddot_d=None):
        """
        Compute control input u.
        
        Args:
            x1 (np.ndarray): State 1 (e.g., position) [n].
            x2 (np.ndarray): State 2 (e.g., velocity) [n].
            x1_d (np.ndarray): Desired State 1 [n].
            x1_dot_d (np.ndarray): Desired Velocity [n].
            x1_ddot_d (np.ndarray, optional): Desired Acceleration [n].
            
        Returns:
            np.ndarray: Control effort u [m].
        """
        x1 = np.array(x1)
        x2 = np.array(x2)
        x1_d = np.array(x1_d)
        x1_dot_d = np.array(x1_dot_d)
        if x1_ddot_d is None:
            x1_ddot_d = np.zeros_like(x1_d)
        else:
            x1_ddot_d = np.array(x1_ddot_d)

        # 1. Step 1: Virtual Control Error e1
        # e1 = x1 - x1_d
        e1 = x1 - x1_d

        # 2. Virtual Control Law alpha
        # alpha = -k1 * e1 + x1_dot_d
        alpha = -self.k1 @ e1 if isinstance(self.k1, np.ndarray) else -self.k1 * e1
        alpha += x1_dot_d

        # 3. Virtual Error e2
        # e2 = x2 - alpha
        e2 = x2 - alpha

        # 4. Compute alpha_dot
        # d/dt(alpha) = -k1 * d/dt(e1) + d/dt(x1_dot_d)
        # d/dt(e1) = x2 - x1_dot_d
        e1_dot = x2 - x1_dot_d
        alpha_dot = -self.k1 @ e1_dot if isinstance(self.k1, np.ndarray) else -self.k1 * e1_dot
        alpha_dot += x1_ddot_d

        # 5. Get System Dynamics
        f_val = self.f(x1, x2)
        g_val = self.g(x1, x2)

        # 6. Control Law Step 2
        # u = g_inv * (-e1 - f + alpha_dot - k2 * e2)
        # Assuming g is invertible (square or full rank)
        inner_term = -e1 - f_val + alpha_dot - (self.k2 @ e2 if isinstance(self.k2, np.ndarray) else self.k2 * e2)

        if np.isscalar(g_val) or g_val.shape == () or g_val.shape == (1,1):
            u = inner_term / g_val
        else:
            # pinv for robustness to non-square systems or singularity
            u = np.linalg.pinv(g_val) @ inner_term

        return u
