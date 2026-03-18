import numpy as np

class INDIController:
    """
    Incremental Nonlinear Dynamic Inversion (INDI) Controller.
    
    Exploits fast control rates and sensor measurements to reduce model dependence.
    System Model:
    x_ddot = f(x, x_dot) + g(x, x_dot) * u
    
    INDI discrete approximation:
    u = u0 + g(x0, x_dot0)^-1 * (v - x_ddot0)
    
    where:
    - u0: Previous control input.
    - x_ddot0: Measured/estimated acceleration at current step.
    - v: Desired acceleration (e.g., from outer-loop PD controller).
    - x0, x_dot0: State measurements at current step.
    """
    def __init__(self, g_func):
        """
        Initialize the INDI Controller.
        
        Args:
            g_func (callable): Function returns g(x, x_dot). matrix/scalar [n x m].
        """
        self.g = g_func

    def compute_control(self, u0, x_ddot0, v, x0, x_dot0):
        """
        Compute control input u.
        
        Args:
            u0 (np.ndarray): Previous control input [m].
            x_ddot0 (np.ndarray): Measured/estimated acceleration [n].
            v (np.ndarray): Desired acceleration [n].
            x0 (np.ndarray): Current state 1 (e.g. position) [n].
            x_dot0 (np.ndarray): Current state 2 (e.g. velocity) [n].
            
        Returns:
            np.ndarray: New control effort u [m].
        """
        u0 = np.array(u0)
        x_ddot0 = np.array(x_ddot0)
        v = np.array(v)
        x0 = np.array(x0)
        x_dot0 = np.array(x_dot0)

        # 1. Get g(x, x_dot)
        g_val = self.g(x0, x_dot0)

        # 2. Compute Increment delta_u
        # delta_u = g^-1 * (v - x_ddot0)
        acc_error = v - x_ddot0

        if np.isscalar(g_val) or g_val.shape == () or g_val.shape == (1,1):
            delta_u = acc_error / g_val
        else:
             # pinv for robustness to non-square systems or singularity
            delta_u = np.linalg.pinv(g_val) @ acc_error

        # 3. New Control
        u = u0 + delta_u
        return u

class INDIOuterLoopPD:
    """
    Helper for outer-loop tracking (computes desired acceleration v).
    """
    def __init__(self, Kp, Kd):
        """
        Args:
            Kp (np.ndarray or float): Proportional gain.
            Kd (np.ndarray or float): Derivative gain.
        """
        self.Kp = Kp
        self.Kd = Kd

    def compute_v(self, x, x_dot, x_d, x_dot_d, x_ddot_d=None):
        x = np.array(x)
        x_dot = np.array(x_dot)
        x_d = np.array(x_d)
        x_dot_d = np.array(x_dot_d)
        if x_ddot_d is None:
            x_ddot_d = np.zeros_like(x_d)
        else:
            x_ddot_d = np.array(x_ddot_d)

        e_x = x - x_d
        e_v = x_dot - x_dot_d

        p_term = self.Kp @ e_x if isinstance(self.Kp, np.ndarray) else self.Kp * e_x
        d_term = self.Kd @ e_v if isinstance(self.Kd, np.ndarray) else self.Kd * e_v

        v = x_ddot_d - p_term - d_term
        return v
