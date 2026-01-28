import numpy as np
from typing import Tuple
from .time_utils import calc_gmst
from gnc_toolkit.utils.state_conversion import rot_z

def eci2ecef(reci, veci, jdut1, dut1=0) -> tuple[np.ndarray, np.ndarray]:
    """Converts ECI to ECEF."""
    
    gmst = calc_gmst(jdut1, dut1)
    R = rot_z(gmst)
    recef = R @ reci
    vecef = R @ veci
    return recef, vecef

def ecef2eci(recef, vecef, jdut1, dut1=0) -> tuple[np.ndarray, np.ndarray]:
    """Converts ECEF to ECI."""
    
    gmst = calc_gmst(jdut1, dut1)
    R = rot_z(-gmst)
    reci = R @ recef
    veci = R @ vecef
    return reci, veci

def eci2lvlh_dcm(reci, veci):
    """Calculates the DCM from ECI to LVLH."""
    z_lvlh = -reci / np.linalg.norm(reci) # NADIR

    h = np.cross(reci, veci)
    y_lvlh = -h / np.linalg.norm(h) # Negative Orbit Normal
    x_lvlh = np.cross(y_lvlh, z_lvlh) # Completes right-handed system
    return np.vstack((x_lvlh, y_lvlh, z_lvlh))

def eci2llh(r_eci, jdut1):
    """Converts ECI to LLH."""
    # WGS84 Ellipsoid Parameters
    a = 6378137.0          # Semi-major axis [m]
    f = 1.0 / 298.257223563 # Flattening
    e2 = f * (2.0 - f)    # Square of eccentricity

    recef, _ = eci2ecef(r_eci, np.zeros(3), jdut1, dut1=0)
    x_ecef, y_ecef, z_ecef = recef

    p = np.sqrt(x_ecef**2 + y_ecef**2)
    lat = np.atan2(z_ecef, p * (1-e2))
    h = 0.0

    for _ in range(5):
        N = a / np.sqrt(1.0 - e2 * np.sin(lat)**2)
        h = p / np.cos(lat) - N
        lat = np.atan2(z_ecef, p * (1.0 - e2 * (N / (N + h))))
    lon = np.atan2(y_ecef, x_ecef)
    return lat, lon, h

def elements2perifocal_dcm(raan, inc, arg_p):
    """Calculates the DCM from perifocal (PQW) to ECI."""
    c_r, s_r = np.cos(raan), np.sin(raan)
    c_i, s_i = np.cos(inc), np.sin(inc)
    c_p, s_p = np.cos(arg_p), np.sin(arg_p)
    
    r11 = c_r*c_p - s_r*c_i*s_p
    r12 = -c_r*s_p - s_r*c_i*c_p
    r13 = s_r*s_i

    r21 = s_r*c_p + c_r*c_i*s_p
    r22 = -s_r*s_p + c_r*c_i*c_p
    r23 = -c_r*s_i

    r31 = s_i*s_p
    r32 = s_i*c_p
    r33 = c_i
    
    return np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ])

def eci2geodetic(r_eci, jd):
    """Converts ECI to Geodetic."""
    x, y, z = r_eci
    r = np.linalg.norm(r_eci)
    
    # Latitude
    lat = np.arcsin(z / r) # radians
    
    # Longitude (inertial)
    lon_i = np.arctan2(y, x) # radians
    
    # GMST Calc
    gmst = calc_gmst(jd)

    lon = lon_i - gmst
    lon = (lon + np.pi) % (2 * np.pi) - np.pi
    
    R_earth = 6378137.0 # meters
    alt = r - R_earth
    
    return np.degrees(lon), np.degrees(lat), alt
