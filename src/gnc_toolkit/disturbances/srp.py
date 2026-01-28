import numpy as np
from gnc_toolkit.environment.solar import Sun

class Canonball:
    """SRP canonball model."""
    def __init__(self):
        self.sun_model = Sun()
        self.P_sun = 4.56e-6 # N/m^2 (Solar radiation pressure at 1 AU)
    
    def get_acceleration(self, r_eci, jd, mass, area, cr):
        """
        Calculate SRP acceleration.
        
        Args:
            r_eci (np.ndarray): Position ECI [m]
            jd (float): Julian Date
            mass (float): Mass [kg]
            area (float): Area [m^2]
            cr (float): Reflectivity coefficient
            
        Returns:
            np.ndarray: Acceleration ECI [m/s^2]
        """
        r_sun = self.sun_model.calculate_sun_eci(jd)
        
        # Vector from Sat to Sun
        sat_to_sun = r_sun - r_eci
        dist_sun = np.linalg.norm(sat_to_sun)
        u_sun = sat_to_sun / dist_sun
        
        # Shadow function (Binary for now: 1=Sun, 0=Eclipse)
        # Simple cylindrical shadow model
        nu = self.check_eclipse(r_eci, r_sun)
        
        acc_mag = nu * self.P_sun * cr * (area / mass)
        
        return -acc_mag * u_sun # force away from sun
    
    def check_eclipse(self, r_sat, r_sun):
        """
        Cylindrical shadow model.
        Returns 1.0 if in sunlight, 0.0 if in shadow.
        """
        # Projection of r_sat onto Sun vector
        u_sun = r_sun / np.linalg.norm(r_sun)
        s = np.dot(r_sat, u_sun)
        
        if s > 0:
            return 1.0
            
        # Perpendicular distance
        r_perp_sq = np.dot(r_sat, r_sat) - s*s
        
        R_earth = 6378137.0
        if r_perp_sq < R_earth**2:
            return 0.0
        return 1.0
