import numpy as np
from typing import Tuple
from .time_utils import calc_gmst
from gnc_toolkit.utils.state_conversion import rot_z, rot_x

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

def eci2eme2000(reci, veci) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts ECI (J2000) to EME2000.
    In many contexts these are treated as identical. 
    Here we treat ECI as Earth-Centered Inertial at epoch of date, 
    and EME2000 as fixed at J2000.
    """
    # For now, identity as a default mapping.
    return reci, veci

def eme20002eci(reci, veci) -> tuple[np.ndarray, np.ndarray]:
    """Converts EME2000 to ECI (J2000)."""
    return reci, veci

def eci2icrf(reci, veci, jd) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts ECI (J2000) to ICRF.
    Includes simplified Precession and Nutation.
    """
    t = (jd - 2451545.0) / 36525.0 # Julian centuries from J2000
    
    # Simplified Precession (Lieske et al., 1977)
    zeta = (2306.2181 * t + 0.30188 * t**2 + 0.017998 * t**3) * (np.pi / 648000.0)
    theta = (2004.3109 * t - 0.42665 * t**2 - 0.041833 * t**3) * (np.pi / 648000.0)
    z = (2306.2181 * t + 1.09468 * t**2 + 0.018203 * t**3) * (np.pi / 648000.0)
    
    P = rot_z(-z) @ rot_y(theta) @ rot_z(-zeta)
    
    r_icrf = P @ reci
    v_icrf = P @ veci
    
    return r_icrf, v_icrf

def icrf2eci(reci, veci, jd) -> tuple[np.ndarray, np.ndarray]:
    """Converts ICRF to ECI (J2000)."""
    t = (jd - 2451545.0) / 36525.0
    zeta = (2306.2181 * t + 0.30188 * t**2 + 0.017998 * t**3) * (np.pi / 648000.0)
    theta = (2004.3109 * t - 0.42665 * t**2 - 0.041833 * t**3) * (np.pi / 648000.0)
    z = (2306.2181 * t + 1.09468 * t**2 + 0.018203 * t**3) * (np.pi / 648000.0)
    
    P_inv = rot_z(zeta) @ rot_y(-theta) @ rot_z(z)
    
    reci_out = P_inv @ reci
    veci_out = P_inv @ veci
    
    return reci_out, veci_out

def rot_y(angle):
    """Rotation matrix for rotation about y-axis."""
    return np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])

def llh2ecef(lat_rad, lon_rad, alt_m) -> np.ndarray:
    """
    Converts Geodetic coordinates (LLH) to ECEF.
    
    Args:
        lat_rad (float): Latitude in radians.
        lon_rad (float): Longitude in radians.
        alt_m (float): Altitude in meters above ellipsoid.
        
    Returns:
        np.ndarray: Position in ECEF [m].
    """
    a = 6378137.0          # Semi-major axis [m]
    f = 1.0 / 298.257223563 # Flattening
    e2 = f * (2.0 - f)    # Square of eccentricity

    N = a / np.sqrt(1.0 - e2 * np.sin(lat_rad)**2)
    
    x = (N + alt_m) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt_m) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1.0 - e2) + alt_m) * np.sin(lat_rad)
    
    return np.array([x, y, z])

