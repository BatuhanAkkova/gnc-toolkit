import numpy as np

def eci2kepler(reci, veci):
    """Converts ECI to Keplerian elements."""
    mu = 398600.4415e9
    r_mag = np.linalg.norm(reci)
    v_mag = np.linalg.norm(veci)
    h = np.cross(reci, veci)
    h_mag = np.linalg.norm(h)
    
    n_vec = np.array([-h[1], h[0], 0]) # Line of Nodes
    n_mag = np.linalg.norm(n_vec)
    
    e_vec = ((v_mag**2 - mu/r_mag)*reci - np.dot(reci, veci)*veci) / mu
    ecc = np.linalg.norm(e_vec) # Eccentricity

    smr = (v_mag**2/2) - (mu/r_mag)
    a = -mu/(2*smr) # Semi-major axis
    p = h_mag**2 / mu # Semi-latus rectum
    
    arglat = 0.0
    truelon = 0.0
    lonper = 0.0
    
    incl = np.arccos(h[2]/h_mag) # Inclination

    if n_mag < 1e-9:
        raan = 0.0
    else:
        raan = np.arccos(np.clip(n_vec[0]/n_mag, -1.0, 1.0)) # Right Ascension of the Ascending Node
        if n_vec[1] < 0:
            raan = 2*np.pi - raan

    if n_mag < 1e-9:
        if np.abs(ecc) > 1e-9:
            # Equatorial elliptical: Argument of Perigee is angle between x-axis and Eccentricity vector
            # Longitude of Perigee = RAAN + argp. RAAN=0. So argp = lonper.
            argp = np.arccos(np.clip(e_vec[0]/ecc, -1.0, 1.0))
            if e_vec[1] < 0:
                argp = 2*np.pi - argp
        else:
            argp = 0.0
    else:
        argp = np.arccos(np.clip(np.dot(n_vec, e_vec)/(n_mag*ecc), -1.0, 1.0)) if ecc > 1e-9 else 0.0
        if e_vec[2] < 0:
            argp = 2*np.pi - argp

    nu = np.arccos(np.clip(np.dot(e_vec, reci)/(ecc*r_mag), -1.0, 1.0)) # True Anomaly
    if np.dot(e_vec, reci) < 0:
        nu = 2*np.pi - nu

    arglat = 0.0
    if n_mag > 1e-9 and r_mag > 1e-9:
        if incl > 1e-6 or abs(incl - np.pi) > 1e-6:
            arglat = np.arccos(np.clip(np.dot(n_vec, reci)/(n_mag*r_mag), -1.0, 1.0)) # Argument of Latitude
            if reci[2] < 0:
                arglat = 2*np.pi - arglat
    
    if ecc > 1e-6 and (incl < 1e-6 or abs(incl - np.pi) < 1e-6):
        lonper = np.arccos(np.clip(e_vec[0]/ecc, -1.0, 1.0)) # Longitude of Perigee
        if e_vec[1] < 0:
            lonper = 2*np.pi - lonper
        if incl > np.pi/2:
            lonper = 2*np.pi - lonper
    
    if r_mag > 1e-6 and ecc < 1e-6 and (incl < 1e-6 or abs(incl - np.pi) < 1e-6):
        truelon = np.arccos(np.clip(reci[0]/r_mag, -1.0, 1.0)) # True Longitude
        if reci[1] < 0:
            truelon = 2*np.pi - truelon
        if incl > np.pi/2:
            truelon = 2*np.pi - truelon

    # Handle circular/equatorial singularities for Mean Anomaly and reconstruction consistency
    if ecc < 1e-6:
        if incl < 1e-6 or abs(incl - np.pi) < 1e-6:
            nu = truelon
        else:
            nu = arglat
    
    if np.isreal(ecc):
        E, M = anomalies(ecc, nu) # Eccentric and Mean Anomaly
    
    return a, ecc, incl, raan, argp, nu, M, E, p, arglat, truelon, lonper 

def kepler2eci(a, ecc, incl, raan, argp, nu):
    """Converts Keplerian elements to ECI."""
    mu = 398600.4415e9
    p = a * (1 - ecc**2)

    r_pqw = np.array([np.cos(nu), np.sin(nu), 0]) * (p / (1 + ecc * np.cos(nu)))
    v_pqw = np.array([-np.sin(nu), ecc + np.cos(nu), 0]) * (np.sqrt(mu / p))

    reci = np.dot(rot_z(-argp), np.dot(rot_x(-incl), np.dot(rot_z(-raan), r_pqw)))
    veci = np.dot(rot_z(-argp), np.dot(rot_x(-incl), np.dot(rot_z(-raan), v_pqw)))

    return reci, veci

def anomalies(ecc, nu):
    """Calculates eccentric and mean anomaly from true anomaly."""
    E, M = np.inf, np.inf
    # Circular 
    if ecc < 1e-6:
        M = nu
        E = nu
    # Elliptical
    elif ecc < 1 - 1e-6:
        s = (np.sqrt(1 - ecc**2) * np.sin(nu)) / (1 + ecc * np.cos(nu))
        c = (ecc + np.cos(nu)) / (1 + ecc * np.cos(nu))
        E = np.arctan2(s,c)
        M = E - ecc * np.sin(E)
    # Hyperbolic
    elif ecc > 1 + 1e-6:
        if ecc > 1 and abs(nu) < np.pi - np.arccos(1 / ecc):
            s = (np.sqrt(ecc**2 - 1) * np.sin(nu)) / (1 + ecc * np.cos(nu))
            E = np.arcsinh(s)
            M = ecc * np.sinh(E) - E
    # Parabolic
    else:
        if abs(nu) < np.radians(168.0):
            E = np.tan(nu/2)
            M = E + (E**3)/3

    if ecc < 1: # E and M should be in (0, 2pi) range
        M = np.fmod(M, 2*np.pi)
        if M < 0:
            M += 2*np.pi
        E = np.fmod(E, 2*np.pi)
    
    return E, M

def rot_x(angle):
    """Rotation matrix for rotation about x-axis."""
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])

def rot_y(angle):
    """Rotation matrix for rotation about y-axis."""
    return np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])

def rot_z(angle):
    """Rotation matrix for rotation about z-axis."""
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
