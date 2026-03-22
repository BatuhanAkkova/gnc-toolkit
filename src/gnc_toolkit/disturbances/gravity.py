"""
Gravitational acceleration models (Two-Body, J2, Harmonics) and Gradient Torques.
"""

import numpy as np
import os
import csv
from gnc_toolkit.utils.frame_conversion import eci2ecef, ecef2eci
from gnc_toolkit.utils.quat_utils import quat_conj, quat_rot

class TwoBodyGravity:
    """
    Two-body gravity model.
    Keplerian orbit, no gravitational perturbation.
    """
    def __init__(self, mu=398600.4418e9):
        self.mu = mu
    
    def get_acceleration(self, r_eci, jd=None):
        """
        Calculate acceleration in ECI frame.
        
        Args:
            r_eci (np.ndarray): Position vector in ECI frame [m]
            jd (float, optional): Julian Date (unused for TwoBody)
            
        Returns:
            np.ndarray: Acceleration vector in ECI frame [m/s^2]
        """
        r_norm = np.linalg.norm(r_eci)
        return -self.mu / r_norm**3 * r_eci

class J2Gravity:
    """
    J2 gravity model.
    Includes J2 perturbation.
    """
    def __init__(self, mu=398600.4418e9, j2=0.001082635855, re=6378137.0):
        self.mu = mu
        self.j2 = j2
        self.re = re
    
    def get_acceleration(self, r_eci, jd=None):
        """
        Calculate J2 acceleration in ECI frame.
        (ignoring precession/nutation for J2 simplified). 
        """
        r_norm = np.linalg.norm(r_eci)
        x, y, z = r_eci
        
        factor = (3/2) * self.j2 * (self.mu / r_norm**2) * (self.re / r_norm)**2
        
        # Zonal harmonic J2 terms
        ax = factor * (x / r_norm) * (5 * (z / r_norm)**2 - 1)
        ay = factor * (y / r_norm) * (5 * (z / r_norm)**2 - 1)
        az = factor * (z / r_norm) * (5 * (z / r_norm)**2 - 3)
        
        return np.array([ax, ay, az]) + TwoBodyGravity(self.mu).get_acceleration(r_eci)

class HarmonicsGravity:
    """
    Spherical Harmonics gravity model.
    EGM 2008 coefficients are used.
    """
    def __init__(self, mu=398600.4418e9, re=6378137.0, n_max=20, m_max=20, file_path=None):
        self.mu = mu
        self.re = re
        self.n_max = n_max
        self.m_max = m_max
        
        self.C = np.zeros((n_max + 1, m_max + 1))
        self.S = np.zeros((n_max + 1, m_max + 1))
        
        if file_path is None:
            # Default path: package_root/egm2008.csv
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_path = os.path.join(base_dir, 'egm2008.csv')
            
        self._load_coefficients(file_path)

    def _load_coefficients(self, file_path):
        if not os.path.exists(file_path):
            print(f"Warning: Gravity coefficient file not found at {file_path}")
            return

        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            for row in reader:
                n, m = int(row[0]), int(row[1])
                c_val, s_val = float(row[2]), float(row[3])
                
                if n <= self.n_max and m <= self.m_max:
                    self.C[n, m] = c_val
                    self.S[n, m] = s_val

    def get_acceleration(self, r_eci, jd):
        """
        Calculate gravitational acceleration including spherical harmonics.
        Uses fully normalized coefficients and recursion.
        
        Args:
            r_eci (np.ndarray): Position ECI [m]
            jd (float): Julian Date
            
        Returns:
            np.ndarray: Acceleration ECI [m/s^2]
        """
        # Coordinate Conversion
        r_ecef, _ = eci2ecef(r_eci, np.zeros(3), jd)
        x, y, z = r_ecef
        r_sq = x*x + y*y + z*z
        r = np.sqrt(r_sq)
        
        # Precompute reused terms
        rho = self.re / r
        sin_lat = z / r
        
        # Normalized Associated Legendre Polynomials Recursive Calculation
        # P[n][m]
        
        P = np.zeros((self.n_max + 2, self.m_max + 2))
        
        # Initial values
        P[0, 0] = 1.0
        P[1, 0] = np.sqrt(3) * sin_lat
        P[1, 1] = np.sqrt(3) * np.sqrt(1 - sin_lat**2) # = sqrt(3)*cos_lat
        
        # Recurrence
        for n in range(2, self.n_max + 1):
            # Zonal (m=0)
            a_n0 = np.sqrt((2*n + 1) / n) * np.sqrt(2*n - 1)
            b_n0 = np.sqrt((2*n + 1) / n) * np.sqrt((n - 1) / (2*n - 3))
            P[n, 0] = a_n0 * sin_lat * P[n-1, 0] - b_n0 * P[n-2, 0]
            
            # Tesseral/Sectorial
            for m in range(1, n + 1):
                if m > self.m_max: break
                
                if n == m: # Sectorial
                    c_nn = np.sqrt((2*n + 1) / (2*n)) 
                    P[n, n] = c_nn * np.sqrt(1 - sin_lat**2) * P[n-1, n-1]
                else: # Tesseral
                    anm = np.sqrt(((2*n - 1) * (2*n + 1)) / ((n - m) * (n + m)))
                    bnm = np.sqrt(((2*n + 1) * (n + m - 1) * (n - m - 1)) / ((2*n - 3) * (n + m) * (n - m)))
                    P[n, m] = anm * sin_lat * P[n-1, m] - bnm * P[n-2, m]

        # Summation
        # Accelerations in ECEF
        ax, ay, az = 0.0, 0.0, 0.0
                
        du_dr = 0 # partial U / partial r
        du_dlat = 0 # partial U / partial lat
        du_dlon = 0 # partial U / partial lon
        
        # Longitude terms
        cos_mlon = np.zeros(self.m_max + 1)
        sin_mlon = np.zeros(self.m_max + 1)
        lambda_lon = np.arctan2(y, x)
        
        for m in range(self.m_max + 1):
            cos_mlon[m] = np.cos(m * lambda_lon)
            sin_mlon[m] = np.sin(m * lambda_lon)
            
        for n in range(2, self.n_max + 1):
            rho_n = rho**n
            
            sum_r, sum_lat, sum_lon = 0, 0, 0
            
            for m in range(0, min(n, self.m_max) + 1):
                C = self.C[n, m]
                S = self.S[n, m]
                
                geo_term = (C * cos_mlon[m] + S * sin_mlon[m])
                
                p_nm = P[n, m]
                
                # Derivative dP/dphi
                if n == m:
                    # Sectorial deriv
                    # P_nn = c * cos^n phi
                    # dP_nn/dphi = -n * tan_phi * P_nn
                    cos_lat_sq = 1 - sin_lat**2
                    if cos_lat_sq > 1e-15:
                        dp_dphi = -n * sin_lat / np.sqrt(cos_lat_sq) * P[n, n]
                    else:
                        dp_dphi = 0.0
                else:
                    anm = np.sqrt((2*n + 1) / (n - m)) * np.sqrt((2*n - 1) / (n + m))
                    
                    term1 = n * sin_lat * P[n, m]
                    term2 = anm * P[n-1, m]
                    cos_lat_sq = 1 - sin_lat**2
                    if cos_lat_sq > 1e-15:
                        dp_dphi = (term1 - term2) / np.sqrt(cos_lat_sq)
                    else:
                        dp_dphi = 0.0 # At poles
                
                # Accumulate
                sum_r += (n + 1) * p_nm * geo_term
                sum_lat += dp_dphi * geo_term
                sum_lon += p_nm * m * (-C * sin_mlon[m] + S * cos_mlon[m])
            
            du_dr   -= (self.mu / r_sq) * rho_n * sum_r
            du_dlat += (self.mu / r)    * rho_n * sum_lat
            du_dlon += (self.mu / r)    * rho_n * sum_lon

        # Rotate to ECEF (Spherical -> Cartesian)
        cos_lat = np.sqrt(1 - sin_lat**2)
        
        ar = du_dr
        alat = (1.0 / r) * du_dlat
        alon = (1.0 / (r * cos_lat)) * du_dlon if cos_lat > 1e-9 else 0
        
        # Rotation matrix from Spherical(r, lat, lon) to ECEF(x,y,z)
        sin_l, cos_l = sin_mlon[1], cos_mlon[1] # m=1 corresponds to lambda
        sin_b, cos_b = sin_lat, cos_lat
        
        ax = ar * (cos_l * cos_b) + alat * (-cos_l * sin_b) + alon * (-sin_l)
        ay = ar * (sin_l * cos_b) + alat * (-sin_l * sin_b) + alon * (cos_l)
        az = ar * (sin_b)         + alat * (cos_b)
        
        acc_ecef = np.array([ax, ay, az])
        
        # Rotate to ECI
        acc_eci, _ = ecef2eci(acc_ecef, np.zeros(3), jd)
        return acc_eci

class GradientTorque:
    """
    Gravity Gradient Torque Calculation.
    """
    def __init__(self, mu=398600.4418e9):
        self.mu = mu

    def gravity_gradient_torque(self, J, r_eci, q_body2eci):
        """
        Calculates Gravity Gradient torque in Body Frame.
        T_gg = 3 * mu / R^5 * (r_body x J * r_body)
        (Note: r_body = -u_nadir_body * R. Formula using unit vector: 3*mu/R^3 * (u x Ju))
        """
        r_mag = np.linalg.norm(r_eci)
        if r_mag == 0: return np.zeros(3)
    
        # Nadir vector (points to Earth Center)
        nadir_eci = -r_eci / r_mag
    
        # Rotate Nadir to Body Frame
        # q rotates Body to ECI. So we need inverse (ECI to Body) = q_conj
        q_eci2body = quat_conj(q_body2eci)
        nadir_body = quat_rot(q_eci2body, nadir_eci)
    
        # J * nadir_body
        j_nadir = J @ nadir_body
    
        # Torque
        # T = 3 * mu / R^3 * (nadir x J*nadir)
        factor = 3 * self.mu / (r_mag**3)
        t_gg = factor * np.cross(nadir_body, j_nadir)
    
        return t_gg

class ThirdBodyGravity:
    """
    Luni-solar third-body gravity model.
    Treats Sun and Moon as point masses.
    """
    def __init__(self, mu_sun=1.32712440018e20, mu_moon=4902.800066e9):
        self.mu_sun = mu_sun
        self.mu_moon = mu_moon
        # Lazy imports to avoid circular dependency
        from gnc_toolkit.environment.solar import Sun
        from gnc_toolkit.environment.moon import Moon
        self.sun_model = Sun()
        self.moon_model = Moon()

    def get_acceleration(self, r_eci, jd):
        """
        Calculate third-body acceleration in ECI frame.
        a = sum( mu_k * ( (s_k - r)/(|s_k - r|^3) - s_k/|s_k|^3 ) )
        """
        r_sun = self.sun_model.calculate_sun_eci(jd) # already in meters
        r_moon = self.moon_model.calculate_moon_eci(jd) # in meters
        
        acc = np.zeros(3)
        
        # Sun contribution
        d_sun = r_sun - r_eci
        acc += self.mu_sun * (d_sun / np.linalg.norm(d_sun)**3 - r_sun / np.linalg.norm(r_sun)**3)
        
        # Moon contribution
        d_moon = r_moon - r_eci
        acc += self.mu_moon * (d_moon / np.linalg.norm(d_moon)**3 - r_moon / np.linalg.norm(r_moon)**3)
        
        return acc

class RelativisticCorrection:
    """
    General Relativistic corrections for gravity.
    Includes Schwarzschild (static) and Lense-Thirring (frame-dragging).
    Reference: IERS Conventions (2010), Chapter 10.
    """
    def __init__(self, mu=398600.4418e9, J_earth=None):
        self.mu = mu
        self.c = 299792458.0 # Speed of light [m/s]
        # Earth angular momentum per unit mass S/m ~ Earth's spin
        # S = I * omega
        if J_earth is None:
            # I_zz ~ 0.3308 * M * R^2
            re = 6378137.0
            omega = 7.292115e-5
            self.S_vec = np.array([0, 0, 0.3308 * re**2 * omega]) # Simplified
        else:
            self.S_vec = J_earth

    def get_acceleration(self, r_eci, v_eci):
        """
        Calculate relativistic acceleration correction.
        """
        r_mag = np.linalg.norm(r_eci)
        v_mag = np.linalg.norm(v_eci)
        
        # Schwarzschild (Standard form)
        # a = (mu/c^2 r^3) * [ (4*mu/r - v^2) * r + 4*(r.v)*v ]
        term1 = 4 * self.mu / r_mag - v_mag**2
        a_sch = (self.mu / (self.c**2 * r_mag**3)) * (term1 * r_eci + 4 * np.dot(r_eci, v_eci) * v_eci)
        
        # Lense-Thirring (Frame dragging)
        # Standard formula: 2 * mu / (c^2 r^3) * [ (3/r^2) (r.S) r - S ] x v
        term_lt = (2 * self.mu / (self.c**2 * r_mag**3)) * ( (3.0/r_mag**2) * np.dot(r_eci, self.S_vec) * r_eci - self.S_vec )
        a_lt = np.cross(term_lt, v_eci)
        
        return a_sch + a_lt

class OceanTidesGravity:
    """
    Ocean Tides Gravity Correction (Simplified Model).
    Implements tidal corrections for n=2, m=1,2 using the four main
    constituents: M2, S2, K1, O1 based on IERS 2010 coefficients.
    Add this acceleration to other gravity models.
    """
    def __init__(self, mu=398600.4418e9, re=6378137.0):
        self.mu = mu
        self.re = re
        
        # Coefficients (Fully Normalized)
        # Values in 10^-11 from IERS 2010
        self.coefs = {
            'M2': {'n': 2, 'm': 2, 'C+': 3.090e-11, 'S+': -1.155e-11},
            'S2': {'n': 2, 'm': 2, 'C+': 0.573e-11, 'S+': -0.134e-11},
            'K1': {'n': 2, 'm': 1, 'C+': -0.155e-11, 'S+': 0.621e-11},
            'O1': {'n': 2, 'm': 1, 'C+': -0.177e-11, 'S+': 0.055e-11}
        }

    def _get_doodson_arguments(self, jd):
        """
        Calculates Doodson arguments for the main constituents.
        """
        from gnc_toolkit.utils.time_utils import calc_gmst
        
        T = (jd - 2451545.0) / 36525.0
        n = jd - 2451545.0
        
        # GMST in radians
        gmst = calc_gmst(jd)
        
        # Mean longitude of the Moon [rad]
        s = np.radians((218.316 + 481267.8813 * T) % 360)
        
        # Mean longitude of the Sun [rad]
        h = np.radians((280.459 + 0.98564736 * n) % 360)
        
        # Doodson Arguments
        theta = {
            'M2': 2 * (gmst + np.pi - s),
            'S2': 2 * (gmst + np.pi - h),
            'K1': gmst + np.pi,
            'O1': gmst + np.pi - 2 * s
        }
        
        return theta

    def get_acceleration(self, r_eci, jd):
        """
        Calculate acceleration due to Ocean Tides in ECI frame.
        
        Args:
            r_eci (np.ndarray): Position ECI [m]
            jd (float): Julian Date
            
        Returns:
            np.ndarray: Acceleration ECI [m/s^2]
        """
        # Get Doodson Arguments
        theta = self._get_doodson_arguments(jd)
        
        # Calculate Delta C and Delta S
        delta_C = np.zeros((3, 3))
        delta_S = np.zeros((3, 3))
        
        for const, vals in self.coefs.items():
            n = vals['n']
            m = vals['m']
            C_plus = vals['C+']
            S_plus = vals['S+']
            arg = theta[const]
            
            delta_C[n, m] += C_plus * np.cos(arg) + S_plus * np.sin(arg)
            delta_S[n, m] += S_plus * np.cos(arg) - C_plus * np.sin(arg)

        # Coordinate Conversion
        r_ecef, _ = eci2ecef(r_eci, np.zeros(3), jd)
        x, y, z = r_ecef
        r_sq = x*x + y*y + z*z
        r_mag = np.sqrt(r_sq)
        
        rho = self.re / r_mag
        sin_lat = z / r_mag
        cos_lat_sq = max(0, 1 - sin_lat**2)
        cos_lat = np.sqrt(cos_lat_sq)
        
        # Evaluate Spherical Harmonics for n=2
        n_max = 2
        m_max = 2
        P = np.zeros((n_max + 1, m_max + 1))
        P[0, 0] = 1.0
        P[1, 0] = np.sqrt(3) * sin_lat
        P[1, 1] = np.sqrt(3) * cos_lat
        
        # n=2 coefficients
        P[2, 0] = np.sqrt(5)/2 * (3 * sin_lat**2 - 1)
        # Sectorial
        P[2, 2] = np.sqrt(5/4) * cos_lat * P[1, 1]
        # Tesseral
        P[2, 1] = np.sqrt(5) * sin_lat * P[1, 1]

        # Derivatives dP/dphi
        dp_dphi = np.zeros((n_max + 1, m_max + 1))
        if cos_lat_sq > 1e-15:
            dp_dphi[1, 0] = np.sqrt(3) * cos_lat # d/dlat sin = cos
            dp_dphi[1, 1] = -np.sqrt(3) * sin_lat # d/dlat cos = -sin
            
            # n=2
            dp_dphi[2, 0] = np.sqrt(5) * 3 * sin_lat * cos_lat
            dp_dphi[2, 1] = np.sqrt(15) * (cos_lat_sq - sin_lat**2)
            dp_dphi[2, 2] = -np.sqrt(15) * sin_lat * cos_lat
        else:
            dp_dphi[:, :] = 0.0

        # Summation
        du_dr = 0
        du_dlat = 0
        du_dlon = 0
        
        lambda_lon = np.arctan2(y, x)
        cos_mlon = np.array([1.0, np.cos(lambda_lon), np.cos(2 * lambda_lon)])
        sin_mlon = np.array([0.0, np.sin(lambda_lon), np.sin(2 * lambda_lon)])

        for n in [2]:
            rho_n = rho**n
            sum_r, sum_lat, sum_lon = 0, 0, 0
            for m in [1, 2]:
                C = delta_C[n, m]
                S = delta_S[n, m]
                
                geo_term = (C * cos_mlon[m] + S * sin_mlon[m])
                p_nm = P[n, m]
                dp_dphi_nm = dp_dphi[n, m]
                
                sum_r += (n + 1) * p_nm * geo_term
                sum_lat += dp_dphi_nm * geo_term
                sum_lon += p_nm * m * (-C * sin_mlon[m] + S * cos_mlon[m])
                
            du_dr   -= (self.mu / r_sq) * rho_n * sum_r
            du_dlat += (self.mu / r_mag)    * rho_n * sum_lat
            du_dlon += (self.mu / r_mag)    * rho_n * sum_lon

        # Rotate to ECEF
        ar = du_dr
        alat = (1.0 / r_mag) * du_dlat
        alon = (1.0 / (r_mag * cos_lat)) * du_dlon if cos_lat > 1e-9 else 0
        
        sin_l, cos_l = sin_mlon[1], cos_mlon[1]
        sin_b, cos_b = sin_lat, cos_lat
        
        ax = ar * (cos_l * cos_b) + alat * (-cos_l * sin_b) + alon * (-sin_l)
        ay = ar * (sin_l * cos_b) + alat * (-sin_l * sin_b) + alon * (cos_l)
        az = ar * (sin_b)         + alat * (cos_b)
        
        acc_ecef = np.array([ax, ay, az])
        acc_eci, _ = ecef2eci(acc_ecef, np.zeros(3), jd)
        return acc_eci