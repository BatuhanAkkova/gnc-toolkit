"""
Spacecraft thermal environment flux models (Solar, Albedo, Earth IR).
"""

import numpy as np

class ThermalEnvironment:
    """
    Models for external thermal fluxes: Solar, Albedo, and Earth IR.
    """
    def __init__(self, albedo_coeff=0.3, earth_ir=230.0):
        """
        Args:
            albedo_coeff (float): Earth's mean albedo coefficient [0-1]
            earth_ir (float): Earth's mean IR flux [W/m^2]
        """
        self.albedo_coeff = albedo_coeff
        self.earth_ir = earth_ir
        self.solar_constant = 1361.0 # W/m^2 at 1 AU

    def get_solar_flux(self, distance_au=1.0):
        """Direct solar flux at distance [W/m^2]."""
        return self.solar_constant / (distance_au**2)

    def get_albedo_flux(self, r_sat, r_sun):
        """
        Calculate albedo flux on a nadir-facing surface.
        Simplified model based on angle between Sat-Sun and Nadir.
        """
        r_sat_norm = r_sat / np.linalg.norm(r_sat)
        r_sun_norm = r_sun / np.linalg.norm(r_sun)
        
        cos_zeta = np.dot(r_sat_norm, r_sun_norm)
        
        if cos_zeta < 0:
            return 0.0 # Night side
            
        # Simplified Lambertian reflection model
        flux = self.solar_constant * self.albedo_coeff * cos_zeta
        return flux

    def get_earth_ir_flux(self):
        """Earth IR flux [W/m^2]."""
        return self.earth_ir
