import numpy as np

class Moon:
    """
    Simplified Lunar Ephemeris Model.
    Based on low-precision formulas (e.g., Vallado).
    """
    def __init__(self):
        self.mu_moon = 4902.800066e9 # m^3/s^2

    def calculate_moon_eci(self, jd):
        """
        Calculates Moon vector in ECI frame.
        
        Args:
            jd (float): Julian Date
            
        Returns:
            np.ndarray: Moon position vector in ECI frame [m]
        """
        T = (jd - 2451545.0) / 36525.0 # Julian centuries since J2000
        
        # Mean longitude of the Moon [deg]
        lam_m = 218.316 + 481267.8813 * T
        
        # Mean anomaly of the Moon [deg]
        M_m = 134.963 + 477198.8676 * T
        
        # Mean anomaly of the Sun [deg]
        M_s = 357.529 + 35999.0503 * T
        
        # Mean elongation of the Moon [deg]
        D = 297.850 + 445267.1115 * T
        
        # Mean latitude of the Moon [deg]
        u = 93.272 + 483202.0175 * T
        
        # Convert to radians
        lam_m_rad = np.radians(lam_m % 360)
        M_m_rad = np.radians(M_m % 360)
        M_s_rad = np.radians(M_s % 360)
        D_rad = np.radians(D % 360)
        u_rad = np.radians(u % 360)
        
        # Simplified series (main terms)
        lon = lam_m + 6.289 * np.sin(M_m_rad) - 1.274 * np.sin(M_m_rad - 2*D_rad) + \
              0.658 * np.sin(2*D_rad) + 0.214 * np.sin(2*M_m_rad)
        lat = 5.128 * np.sin(u_rad) + 0.280 * np.sin(u_rad + M_m_rad) + \
              0.277 * np.sin(M_m_rad - u_rad) + 0.173 * np.sin(u_rad - 2*D_rad)
        parallax = 0.9508 + 0.0518 * np.cos(M_m_rad) + 0.0095 * np.cos(M_m_rad - 2*D_rad) + \
                   0.0078 * np.cos(2*D_rad) + 0.0028 * np.cos(2*M_m_rad)
        
        # Distance [km]
        r_mag = 6378.137 / np.sin(np.radians(parallax))
        
        lon_rad = np.radians(lon % 360)
        lat_rad = np.radians(lat % 360)
        
        # Convert Ecliptic to Equatorial (Obliquity epsilon ~ 23.439 deg)
        eps = np.radians(23.439291)
        
        x = r_mag * np.cos(lat_rad) * np.cos(lon_rad)
        y = r_mag * (np.cos(eps) * np.cos(lat_rad) * np.sin(lon_rad) - np.sin(eps) * np.sin(lat_rad))
        z = r_mag * (np.sin(eps) * np.cos(lat_rad) * np.sin(lon_rad) + np.cos(eps) * np.sin(lat_rad))
        
        return np.array([x, y, z]) * 1000.0 # Convert to meters
