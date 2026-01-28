import numpy as np
from gnc_toolkit.utils.state_to_elements import eci2kepler, kepler2eci

def point_mass_grav(r):
    """Calculates the gravitational acceleration as point mass."""
    mu = 398600.4418e9 # m^3/s^2
    r_mag = np.linalg.norm(r)
    
    a = -mu * r / r_mag**3
    return a

def j2_grav(r, v, dt, ndot=0, nddot=0):
    """Calculates the gravitational acceleration accounting for J2."""
    mu = 398600.4418e9 # m^3/s^2
    j2 = 1.08262668e-3
    R_e = 6378137 # m

    a, ecc, incl, raan, argp, nu, _, _, p, _, _, _ = eci2kepler(r, v)
    if a < 0: n = None 
    else: n = np.sqrt(mu / (a**3))

    if n is None: return np.zeros(3), np.zeros(3)

    j2_eff = (n * 1.5 * R_e**2 * j2) / (p**2)
    
    raandot = -j2_eff * np.cos(incl)
    raan += raandot * dt
    
    argpdot = j2_eff * (2 - 2.5 * np.sin(incl)**2)
    argp += argpdot * dt
    
    mdot = n
    nu += mdot * dt + ndot * dt**2 + nddot * dt**3
    
    a -= 2 * ndot * dt * a / (3 * n)
    ecc -= 2 * (1 - ecc) * ndot * dt / (3 * n)
    
    return kepler2eci(a, ecc, incl, raan, argp, nu)
    
