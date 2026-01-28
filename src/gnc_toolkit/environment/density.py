import numpy as np
from gnc_toolkit.utils.frame_conversion import eci2geodetic, eci2llh
from gnc_toolkit.utils.time_utils import calc_jd
from gnc_toolkit.environment.solar import Sun
import pymsis

class Exponential:
    def __init__(self, rho0=1.225, h0=0.0, H=8.5):
        """Simple Exponential Density Model."""
        self.rho0 = rho0
        self.h0 = h0
        self.H = H
        self.R_earth = 6378.137 # km

    def get_density(self, r_eci, jd):
        _, _, h = eci2geodetic(r_eci, jd)
        
        if h < 0:
            return self.rho0
            
        # Convert h to km for scale height calculation
        h_km = h / 1000.0
        rho = self.rho0 * np.exp(-(h_km - self.h0) / self.H)
        return rho

class HarrisPriester:
    def __init__(self, lag=30.0):
        """Harris Priester Density Model"""
        self.lag = np.radians(lag)
        self.sun_model = Sun()

    def get_density(self, r_eci, jd):
        r_sun = self.sun_model.calculate_sun_eci(jd)
        sun = r_sun / np.linalg.norm(r_sun) # Unit vector

        apex = np.array([
            sun[0] * np.cos(self.lag) - sun[1] * np.sin(self.lag),
            sun[0] * np.sin(self.lag) + sun[1] * np.cos(self.lag),
            sun[2]
        ]) # Apex of the diurnal bulge

        r_unit = r_eci / np.linalg.norm(r_eci)
        cos_psi = np.dot(r_unit, apex) # Angle between satellite and apex

        n = 2 # 2 for low inclination, 6 for polar orbit
        cos_term = np.abs(np.cos(np.arccos(cos_psi) / 2.0)) ** n

        # Look up table for mean solar flux (F10.7=150)
        # altitude (m), min_density (kg/m^3), max_density (kg/m^3)
        rho_mod = [
            (100000.0, 4.974e-07, 4.974e-07),
            (120000.0, 2.490e-08, 2.490e-08),
            (130000.0, 8.377e-09, 8.710e-09),
            (140000.0, 3.899e-09, 4.059e-09),
            (150000.0, 2.122e-09, 2.215e-09),
            (160000.0, 1.263e-09, 1.344e-09),
            (170000.0, 8.008e-10, 8.758e-10),
            (180000.0, 5.283e-10, 6.010e-10),
            (190000.0, 3.617e-10, 4.297e-10),
            (200000.0, 2.557e-10, 3.162e-10),
            (210000.0, 1.839e-10, 2.396e-10),
            (220000.0, 1.341e-10, 1.853e-10),
            (230000.0, 9.949e-11, 1.455e-10),
            (240000.0, 7.488e-11, 1.157e-10),
            (250000.0, 5.709e-11, 9.308e-11),
            (260000.0, 4.403e-11, 7.555e-11),
            (270000.0, 3.430e-11, 6.182e-11),
            (280000.0, 2.697e-11, 5.095e-11),
            (290000.0, 2.139e-11, 4.226e-11),
            (300000.0, 1.708e-11, 3.526e-11),
            (320000.0, 1.099e-11, 2.511e-11),
            (340000.0, 7.214e-12, 1.819e-11),
            (360000.0, 4.824e-12, 1.337e-11),
            (380000.0, 3.274e-12, 9.955e-12),
            (400000.0, 2.249e-12, 7.492e-12),
            (420000.0, 1.558e-12, 5.684e-12),
            (440000.0, 1.091e-12, 4.355e-12),
            (460000.0, 7.701e-13, 3.362e-12),
            (480000.0, 5.474e-13, 2.612e-12),
            (500000.0, 3.916e-13, 2.042e-12),
            (520000.0, 2.819e-13, 1.605e-12),
            (540000.0, 2.042e-13, 1.267e-12),
            (560000.0, 1.488e-13, 1.005e-12),
            (580000.0, 1.092e-13, 7.997e-13),
            (600000.0, 8.070e-14, 6.390e-13),
            (620000.0, 6.012e-14, 5.123e-13),
            (640000.0, 4.519e-14, 4.121e-13),
            (660000.0, 3.430e-14, 3.325e-13),
            (680000.0, 2.632e-14, 2.691e-13),
            (700000.0, 2.043e-14, 2.185e-13),
            (720000.0, 1.607e-14, 1.779e-13),
            (740000.0, 1.281e-14, 1.452e-13),
            (760000.0, 1.036e-14, 1.190e-13),
            (780000.0, 8.496e-15, 9.776e-14),
            (800000.0, 7.069e-15, 8.059e-14),
            (840000.0, 4.680e-15, 5.741e-14),
            (880000.0, 3.200e-15, 4.210e-14),
            (920000.0, 2.210e-15, 3.130e-14),
            (960000.0, 1.560e-15, 2.360e-14),
            (1000000.0, 1.150e-15, 1.810e-14)
        ]

        _, _, h = eci2llh(r_eci, jd) # Altitude [m]

        # Edge Cases
        if h < rho_mod[0][0]: return rho_mod[0][1]
        elif h > rho_mod[-1][0]: return rho_mod[-1][1]

        i = 0
        while i < len(rho_mod) - 2 and h > rho_mod[i+1][0]:
            i += 1

        # Linear interpolation in Log-Density
        h1 = rho_mod[i][0]
        rho_min1 = rho_mod[i][1]
        rho_max1 = rho_mod[i][2]
        
        h2 = rho_mod[i+1][0]
        rho_min2 = rho_mod[i+1][1]
        rho_max2 = rho_mod[i+1][2]
        
        frac = (h - h1) / (h2 - h1)
        
        rho_min = np.exp(np.log(rho_min1) + frac * (np.log(rho_min2) - np.log(rho_min1)))
        rho_max = np.exp(np.log(rho_max1) + frac * (np.log(rho_max2) - np.log(rho_max1)))

        return rho_min + (rho_max - rho_min) * cos_term

class NRLMSISE00:
    """NRLMSISE-00 atmospheric model using pymsis."""
    
    def __init__(self):
        pass

    def get_density(self, r_eci, date):
        """
        Get total mass density [kg/m^3].
        Args:
            r_eci (np.ndarray): Position vector in ECI frame [km].
            date (datetime): Current time.
        """
        jd, jdfrac = calc_jd(date.year, date.month, date.day, date.hour, date.minute, date.second)
        jd += jdfrac
        lon, lat, alt = eci2geodetic(r_eci, jd)

        output = pymsis.calculate(date, lon, lat, alt)
        
        output = np.squeeze(output)
        if output.ndim == 0:
            rho = float(output)
        else:
            rho = float(output[0]) # Total mass density
        
        return rho
