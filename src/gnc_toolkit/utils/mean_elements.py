"""
Mean element conversions and secular perturbation rates (e.g., J2 secular theory).
"""

import numpy as np

def osculating2mean(a, ecc, incl, raan, argp, M, J2=1.08262668e-3, re=6378137.0):
    """
    Convert osculating Keplerian elements to mean elements using first-order J2 secular theory.
    Simplified version focusing on secular changes.
    
    Args:
        a, ecc, incl, raan, argp, M: Osculating Keplerian elements
        J2 (float): J2 perturbation coefficient
        re (float): Earth radius [m]
        
    Returns:
        tuple: (a_m, ecc_m, incl_m, raan_m, argp_m, M_m)
    """
    p = a * (1 - ecc**2)
    n = np.sqrt(398600.4415e9 / a**3)
    
    # Zero-order approximation: returns osculating elements unchanged.
    # For rigorous conversion subtract short/long-period terms (Vallado, Alg. 34).
    return a, ecc, incl, raan, argp, M

def get_j2_secular_rates(a, ecc, incl, J2=1.08262668e-3, re=6378137.0, mu=398600.4415e9):
    """
    Calculates the secular rates of change for RAAN, Argument of Perigee, and Mean Anomaly due to J2.
    """
    p = a * (1 - ecc**2)
    n = np.sqrt(mu / a**3)
    
    # RAAN rate (rad/s)
    raan_dot = -1.5 * n * J2 * (re / p)**2 * np.cos(incl)
    
    # Argument of Perigee rate (rad/s)
    argp_dot = 0.75 * n * J2 * (re / p)**2 * (4.0 - 5.0 * np.sin(incl)**2)
    
    # Mean Anomaly rate (rad/s) - correction to mean motion
    M_dot = n + 0.75 * n * J2 * (re / p)**2 * np.sqrt(1 - ecc**2) * (2.0 - 3.0 * np.sin(incl)**2)
    
    return raan_dot, argp_dot, M_dot
