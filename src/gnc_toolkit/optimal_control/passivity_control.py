"""
Passivity-Based Controller for Euler-Lagrange mechanical systems.
"""

import numpy as np

class PassivityBasedController:
    """
    Passivity-Based Controller for Euler-Lagrange mechanical systems.
    
    Exploits the energy properties and skew-symmetric property of (M_dot - 2C).
    Commonly known as the Slotine & Li controller.
    System: M(q) * q_ddot + C(q, q_dot) * q_dot + G(q) = u
    """
    def __init__(self, M_func, C_func, G_func, K_d, Lambda):
        """
        Initialize the Passivity-Based Controller.
        
        Args:
            M_func (callable): Function returns Inertia matrix M(q) [n x n].
            C_func (callable): Function returns Coriolis matrix C(q, q_dot) [n x n].
            G_func (callable): Function returns Gravity vector G(q) [n].
            K_d (np.ndarray): Derivative gain matrix [n x n] or scalar.
            Lambda (np.ndarray): Proportional error weight matrix [n x n] or scalar.
        """
        self.M = M_func
        self.C = C_func
        self.G = G_func
        self.K_d = K_d
        self.Lambda = Lambda

    def compute_control(self, q, q_dot, q_d, q_dot_d, q_ddot_d=None):
        """
        Compute control input.
        
        Args:
            q (np.ndarray): Current positions [n].
            q_dot (np.ndarray): Current velocities [n].
            q_d (np.ndarray): Desired positions [n].
            q_dot_d (np.ndarray): Desired velocities [n].
            q_ddot_d (np.ndarray, optional): Desired accelerations [n]. Defaults to zeros.
            
        Returns:
            np.ndarray: Control effort u [n].
        """
        q = np.array(q)
        q_dot = np.array(q_dot)
        q_d = np.array(q_d)
        q_dot_d = np.array(q_dot_d)
        if q_ddot_d is None:
            q_ddot_d = np.zeros_like(q_d)
        else:
            q_ddot_d = np.array(q_ddot_d)

        # 1. Compute Errors
        e_q = q - q_d
        e_q_dot = q_dot - q_dot_d

        # 2. Reference Velocity v_r
        # v_r = q_dot_d - Lambda * e_q
        v_r = q_dot_d - self.Lambda @ e_q if isinstance(self.Lambda, np.ndarray) else q_dot_d - self.Lambda * e_q

        # 3. Reference Acceleration v_r_dot
        # v_r_dot = q_ddot_d - Lambda * e_q_dot
        v_r_dot = q_ddot_d - self.Lambda @ e_q_dot if isinstance(self.Lambda, np.ndarray) else q_ddot_d - self.Lambda * e_q_dot

        # 4. Sliding Surface / Velocity Error s
        # s = q_dot - v_r = e_q_dot + Lambda * e_q
        s = q_dot - v_r

        # 5. Get System Dynamics Matrices
        M_mat = self.M(q)
        C_mat = self.C(q, q_dot)
        G_vec = self.G(q)

        # 6. Control Law
        # u = M * v_r_dot + C * v_r + G - K_d * s
        feedforward = M_mat @ v_r_dot + C_mat @ v_r + G_vec
        damping = self.K_d @ s if isinstance(self.K_d, np.ndarray) else self.K_d * s
        
        u = feedforward - damping
        return u
