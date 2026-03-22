"""
Atmospheric wind and earth co-rotation modeling.
"""

import numpy as np
from gnc_toolkit.utils.frame_conversion import eci2ecef, ecef2eci

class AtmosphereCoRotation:
    """
    Calculates atmospheric wind velocity assuming co-rotation with Earth.
    """
    def __init__(self, omega_earth=7.2921151467e-5):
        """
        Initialize with Earth's angular velocity [rad/s].
        """
        self.omega_earth = np.array([0, 0, omega_earth])

    def get_wind_velocity(self, r_eci, jd):
        """
        Calculate wind velocity in ECI frame.
        v_wind = omega x r
        
        Args:
            r_eci (np.ndarray): Position vector in ECI frame [m]
            jd (float): Julian Date
            
        Returns:
            np.ndarray: Wind velocity vector in ECI [m/s]
        """
        v_wind = np.cross(self.omega_earth, r_eci)
        return v_wind

    def get_relative_velocity(self, r_eci, v_eci, jd):
        """
        Calculate spacecraft velocity relative to the atmosphere.
        v_rel = v_sc - v_wind
        """
        v_wind = self.get_wind_velocity(r_eci, jd)
        return v_eci - v_wind
