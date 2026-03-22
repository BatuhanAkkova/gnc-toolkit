"""
Navigation using Line-of-Sight (LOS) measurements (unit vectors).
"""

import numpy as np
from .orbit_determination import OrbitDeterminationEKF

class AngleOnlyNavigation(OrbitDeterminationEKF):
    """
    Navigation using Line-of-Sight (LOS) measurements (unit vectors).
    Inherits orbital dynamics from OrbitDeterminationEKF.
    """
    def update_unit_vector(self, u_meas, target_pos_eci):
        """
        Update state using a unit vector measurement to a target.
        
        Args:
            u_meas (np.ndarray): Measured unit vector [ux, uy, uz] in ECI.
            target_pos_eci (np.ndarray): Position of the target being tracked in ECI [m].
        """
        def hx(x):
            r = x[:3]
            rel_r = target_pos_eci - r
            rel_r_mag = np.linalg.norm(rel_r)
            if rel_r_mag < 1e-3:
                return np.zeros(3)
            return rel_r / rel_r_mag
            
        def H_jac(x):
            r = x[:3]
            rel_r = target_pos_eci - r
            rel_r_mag = np.linalg.norm(rel_r)
            
            if rel_r_mag < 1e-3:
                return np.zeros((3, 6))
                
            u = rel_r / rel_r_mag
            H_rel = (np.eye(3) - np.outer(u, u)) / rel_r_mag
            
            H = np.zeros((3, 6))
            H[:, :3] = -H_rel
            return H
            
        self.ekf.update(u_meas, hx, H_jac)
