import numpy as np
import datetime

class SpaceWeather:
    """
    Model for managing space weather indices like F10.7, Ap, and Kp.
    Used for atmospheric density and radiation models.
    """
    def __init__(self, f107=150.0, f107_avg=150.0, ap=15.0):
        """
        Initialize with default solar flux and geomagnetic indices.
        
        Args:
            f107 (float): Daily solar flux index (10.7 cm) [sfu]
            f107_avg (float): 81-day average solar flux index [sfu]
            ap (float): Planetary equivalent amplitude index
        """
        self.f107 = f107
        self.f107_avg = f107_avg
        self.ap = ap
        self.kp = self._ap_to_kp(ap)

    def _ap_to_kp(self, ap):
        """Approximate conversion from Ap to Kp index."""
        if ap <= 0: return 0.0
        return 0.5 * np.log2(ap / 2.0 + 1.0) * 3.0 # Very rough approximation

    def get_indices(self, date=None):
        """
        Get indices for a specific date. 
        Currently returns static values, but can be extended for database lookup.
        """
        return {
            'f107': self.f107,
            'f107_avg': self.f107_avg,
            'ap': self.ap,
            'kp': self.kp
        }

    def set_solar_flux(self, f107, f107_avg=None):
        self.f107 = f107
        if f107_avg is not None:
            self.f107_avg = f107_avg
        else:
            self.f107_avg = f107
