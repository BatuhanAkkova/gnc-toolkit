
import numpy as np

class Sun:
    def __init__(self):
        pass

    def calculate_sun_eci(self, jd):
        """
        Calculates Sun vector in ECI frame.
        
        Args:
            jd (float): Julian Date
            
        Returns:
            np.array: Sun vector in ECI frame (km).
        """
        n = jd - 2451545.0 # Number of days since J2000

        # Mean anomaly of the Sun
        g = 357.529 + 0.98560028 * n
        g_rad = np.radians(g)
        
        # Mean longitude
        q = 280.459 + 0.98564736 * n
        q_rad = np.radians(q)
        
        # Ecliptic longitude
        L = q + 1.915 * np.sin(g_rad) + 0.020 * np.sin(2 * g_rad)
        L_rad = np.radians(L)
        
        # Obliquity of the ecliptic
        e = 23.439 - 0.00000036 * n
        e_rad = np.radians(e)
        
        # Distance is approx 1 AU
        au = 149597870700.0 # meters (approx)
        
        x = np.cos(L_rad)
        y = np.cos(e_rad) * np.sin(L_rad)
        z = np.sin(e_rad) * np.sin(L_rad)
        
        r_sun = np.array([x, y, z]) * au
        
        return r_sun
