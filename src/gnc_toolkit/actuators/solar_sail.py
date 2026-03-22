"""
Solar Sail force model based on Solar Radiation Pressure (SRP).
"""

import numpy as np
from gnc_toolkit.actuators.actuator import Actuator

class SolarSail(Actuator):
    """
    Solar Sail Actuator.
    Models force produced by Solar Radiation Pressure (SRP) on a flat sail.
    """
    def __init__(self, area, reflectivity=0.9, specular_reflect_coeff=0.9, name="SolarSail"):
        """
        Args:
            area (float): Surface area [m^2].
            reflectivity (float): Total reflectivity coefficient [0 to 1].
            specular_reflect_coeff (float): Fraction of reflection that is specular [0 to 1].
        """
        super().__init__(name=name)
        self.area = area
        self.rho = reflectivity
        self.s_coeff = specular_reflect_coeff
        self.P0 = 4.56e-6  # Solar radiation pressure at 1 AU [N/m^2]

    def calculate_force(self, sun_unit_vec, normal_unit_vec, distance_au=1.0):
        """
        Calculate SRP force on the sail.
        
        Args:
            sun_unit_vec (np.array): Unit vector from sail to Sun.
            normal_unit_vec (np.array): Unit vector normal to sail surface.
            distance_au (float): Distance from Sun [AU].
            
        Returns:
            np.array: Force vector [N].
        """
        u_sun = sun_unit_vec / np.linalg.norm(sun_unit_vec)
        n = normal_unit_vec / np.linalg.norm(normal_unit_vec)
        
        cos_theta = np.dot(u_sun, n)
        if cos_theta < 0:
            # Sun is behind the sail, no force (assuming opaque back for now)
            return np.zeros(3)
        
        # P = P0 / d^2
        P = self.P0 / (distance_au**2)
        
        # Flat-plate SRP: F = P*A*cos(theta)*[(1-rho_s)*u_sun + 2*(rho_s*cos(theta) + rho_d/3)*n]
        rho_s = self.rho * self.s_coeff
        rho_d = self.rho * (1 - self.s_coeff)
        
        force = P * self.area * cos_theta * (
            (1 - rho_s) * u_sun + 
            (2 * rho_s * cos_theta + (2/3) * rho_d) * n
        )
        
        return force

    def command(self, normal_cmd, **kwargs):
        """
        Calculate force based on commanded normal vector and Sun position.
        
        Args:
            normal_cmd (np.array): Commanded normal vector (towards Sun-ish).
            kwargs: Must include 'sun_vec' (relative unit vector).
            
        Returns:
            np.array: Force vector [N].
        """
        sun_vec = kwargs.get('sun_vec', np.array([1, 0, 0]))
        dist = kwargs.get('distance_au', 1.0)
        return self.calculate_force(sun_vec, normal_cmd, distance_au=dist)
