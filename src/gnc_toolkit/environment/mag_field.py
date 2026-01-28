import ppigrf
import numpy as np

def igrf_field(lat, lon, alt, time):
    """Get the IGRF magnetic field at a given location and time."""
    if ppigrf is None:
        raise ImportError("ppigrf not installed or data files missing.")
    return np.array(ppigrf.igrf(lon, lat, alt, time))

def tilted_dipole_field(r_ecef):
    """Calculates the magnetic field vector at a given position using the tilted dipole model."""
    B_0 = 3.12e-5 # Magnetic field strength at equator [T]
    Re = 6371200 # Earth's radius [m]

    # Assume constant
    lat = np.deg2rad(78.3) # Magnetic dip latitude [rad]
    lon = np.deg2rad(-71.8) # Magnetic longitude [rad]

    theta = np.pi/2 - lat
    phi = lon

    moment_u = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ]) # Dipole Moment unit vector

    k = B_0 * Re**3 # Magnitude Factor

    r_mag = np.linalg.norm(r_ecef)
    r_u = r_ecef / r_mag

    B_ecef = (k / r_mag**3) * (3 * np.dot(moment_u, r_u) * r_u - moment_u)

    return -B_ecef # Negative sign because the magnetic field points opposite to the dipole moment
