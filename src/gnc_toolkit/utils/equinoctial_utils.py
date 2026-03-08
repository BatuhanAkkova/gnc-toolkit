import numpy as np
from .state_to_elements import anomalies, kepler2eci, eci2kepler

def kepler2equinoctial(a, ecc, incl, raan, argp, M):
    """
    Converts Keplerian elements to Equinoctial elements.
    
    Equinoctial elements (a, h, k, p, q, lambda) are non-singular for 
    zero eccentricity and zero/90-deg inclination.
    
    Args:
        a (float): Semi-major axis
        ecc (float): Eccentricity
        incl (float): Inclination [rad]
        raan (float): RAAN [rad]
        argp (float): Argument of perigee [rad]
        M (float): Mean anomaly [rad]
        
    Returns:
        tuple: (a, h, k, p, q, l_mean)
    """
    h = ecc * np.sin(raan + argp)
    k = ecc * np.cos(raan + argp)
    p = np.tan(incl / 2.0) * np.sin(raan)
    q = np.tan(incl / 2.0) * np.cos(raan)
    l_mean = raan + argp + M
    
    # Wrap l_mean to [0, 2pi]
    l_mean = np.mod(l_mean, 2 * np.pi)
    
    return a, h, k, p, q, l_mean

def equinoctial2kepler(a, h, k, p, q, l_mean):
    """
    Converts Equinoctial elements to Keplerian elements.
    """
    ecc = np.sqrt(h**2 + k**2)
    incl = 2.0 * np.arctan(np.sqrt(p**2 + q**2))
    raan = np.arctan2(p, q)
    argp = np.arctan2(h, k) - raan
    
    # Solve for F (Equinoctial Eccentric Anomaly)
    F = solve_kepler_equinoctial(l_mean, h, k)
    
    # True anomaly nu from F, h, k
    sinF = np.sin(F)
    cosF = np.cos(F)
    beta = 1.0 / (1.0 + np.sqrt(1.0 - h**2 - k**2))
    
    # Components in the equinoctial frame
    X = a * ((1.0 - h**2 * beta) * cosF + h * k * beta * sinF - k)
    Y = a * ((1.0 - k**2 * beta) * sinF + h * k * beta * cosF - h)
    
    nu_eq = np.arctan2(Y, X)
    # This nu_eq is the angle in the orbital plane from the equinoctial x-axis.
    # To get Keplerian nu (from perigee):
    nu = nu_eq - (raan + argp)
    
    raan = np.mod(raan, 2 * np.pi)
    argp = np.mod(argp, 2 * np.pi)
    nu = np.mod(nu, 2 * np.pi)
    
    # M from F
    M = F - k*np.sin(F) + h*np.cos(F) # This is not quite right for M, wait.
    # From Vallado: lambda = F + h*cos(F) - k*sin(F). 
    # M = lambda - raan - argp
    M = l_mean - raan - argp
    M = np.mod(M, 2 * np.pi)
    
    return a, ecc, incl, raan, argp, nu, M

def solve_kepler_equinoctial(l_mean, h, k):
    """
    Solves Kepler's equation for equinoctial elements to find F (Equinoctial Eccentric Anomaly).
    lambda = F + h*cos(F) - k*sin(F)
    """
    F = l_mean
    for _ in range(10):
        f_val = F + h * np.cos(F) - k * np.sin(F) - l_mean
        df_val = 1.0 - h * np.sin(F) - k * np.cos(F)
        F = F - f_val / df_val
        if abs(f_val) < 1e-12:
            break
    return F

def equinoctial2eci(a, h, k, p, q, l_mean, mu=398600.4415e9):
    """
    Direct conversion from Equinoctial to ECI position and velocity.
    """
    F = solve_kepler_equinoctial(l_mean, h, k)
    
    sinF = np.sin(F)
    cosF = np.cos(F)
    
    beta = 1.0 / (1.0 + np.sqrt(1.0 - h**2 - k**2))
    r = a * (1.0 - k * cosF - h * sinF)
    
    X = a * ((1.0 - h**2 * beta) * cosF + h * k * beta * sinF - k)
    Y = a * ((1.0 - k**2 * beta) * sinF + h * k * beta * cosF - h)
    
    X_dot = (np.sqrt(mu * a) / r) * (h * k * beta * cosF - (1.0 - h**2 * beta) * sinF)
    Y_dot = (np.sqrt(mu * a) / r) * ((1.0 - k**2 * beta) * cosF - h * k * beta * sinF)
    
    # Transformation to ECI
    inv_sq_pq = 1.0 / (1.0 + p**2 + q**2)
    f_vec = np.array([1.0 - p**2 + q**2, 2.0 * p * q, -2.0 * p]) * inv_sq_pq
    g_vec = np.array([2.0 * p * q, 1.0 + p**2 - q**2, 2.0 * q]) * inv_sq_pq
    
    reci = X * f_vec + Y * g_vec
    veci = X_dot * f_vec + Y_dot * g_vec
    
    return reci, veci
