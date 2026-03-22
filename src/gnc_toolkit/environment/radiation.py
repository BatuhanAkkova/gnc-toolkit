"""
Radiation environment models for TID and SEU rates estimation.
"""

import numpy as np

class RadiationModel:
    """
    Basic radiation environment models for spacecraft.
    """
    def __init__(self):
        pass

    def estimate_tid(self, altitude_km, inclination_deg, duration_days):
        """
        Estimate Total Ionising Dose (TID) in kRad(Si).
        Rough parametric model for LEO.
        """
        # Simplified parametric model for 0.1" Al shielding
        base_rate = 1.0e-4 # kRad/day base for low LEO
        
        # Altitude factor
        alt_factor = np.exp((altitude_km - 400) / 500)
        
        # Inclination factor
        inc_factor = 1.0 + 0.5 * np.sin(np.radians(inclination_deg))
        
        tid = base_rate * alt_factor * inc_factor * duration_days
        return tid

    def estimate_seu_rate(self, altitude_km, device_cross_section=1.0e-12):
        """
        Estimate Single Event Upset (SEU) rate.
        
        Args:
            altitude_km (float): Orbit altitude
            device_cross_section (float): Sensitive area [cm^2/bit]
            
        Returns:
            float: SEUs per bit-day
        """
        # Simplified proton flux model
        flux = 100 * np.exp((altitude_km - 400) / 600) # protons/cm^2/s > 10MeV
        rate = flux * device_cross_section * 86400 # per bit-day
        return rate
