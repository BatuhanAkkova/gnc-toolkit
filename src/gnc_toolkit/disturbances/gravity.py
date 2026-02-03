import numpy as np
import os
import csv
from gnc_toolkit.utils.frame_conversion import eci2ecef, ecef2eci

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
            # Assuming src/disturbances/gravity.py -> ../egm2008.csv
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
                    anm = np.sqrt((2*n + 1) / (n - m)) * np.sqrt((2*n - 1) / (n + m))
                    bnm = np.sqrt((2*n + 1) / (n - m)) * np.sqrt((n + m - 1) / (n - m - 1)) * np.sqrt((n - m - 1) / (2*n - 3))
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
                     dp_dphi = -n * sin_lat / np.sqrt(1 - sin_lat**2) * P[n, n] if abs(sin_lat) < 1 else 0
                else:
                    # Recursive deriv                    
                    #factor_nm = np.sqrt((2*n+1)/(2*n-1) * (n-m)/(n+m)) if (n+m) > 0 else 0
                    
                    anm = np.sqrt((2*n + 1) / (n - m)) * np.sqrt((2*n - 1) / (n + m))
                    
                    term1 = n * sin_lat * P[n, m]
                    term2 = anm * P[n-1, m]
                    dp_dphi = (term1 - term2) / np.sqrt(1 - sin_lat**2)
                
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
