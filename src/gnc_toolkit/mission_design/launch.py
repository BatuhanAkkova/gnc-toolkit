import numpy as np
from gnc_toolkit.utils.frame_conversion import llh2ecef, ecef2eci

def calculate_launch_windows(jd_start, jd_end, inc_deg, raan_deg, lat_deg, lon_deg, step_sec=60):
    """
    Calculates launch windows by finding times when the launch site intersects the target orbit plane.
    
    Args:
        jd_start (float): Start Julian Date.
        jd_end (float): End Julian Date.
        inc_deg (float): Target Inclination [deg].
        raan_deg (float): Target RAAN [deg].
        lat_deg (float): Launch site Latitude [deg].
        lon_deg (float): Launch site Longitude [deg].
        step_sec (float): Search step size [s].
        
    Returns:
        dict: list of dicts with 'jd', 'type' (Ascending/Descending), and 'azimuth_deg'.
    """
    R_earth = 6378137.0
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    
    # Orbit plane normal in ECI
    inc_rad = np.radians(inc_deg)
    raan_rad = np.radians(raan_deg)
    # N = [sin(inc) * sin(raan), -sin(inc) * cos(raan), cos(inc)]
    # Ascending node is along [cos(raan), sin(raan), 0].
    # Component along X is cos(raan).
    # Component along Y is sin(raan).
    # Normal vector is cross product of vector to ascending node and direction in plane.
    # Standard: N = [sin(inc)*sin(raan), -sin(inc)*cos(raan), cos(inc)]
    N_orbit = np.array([
        np.sin(inc_rad) * np.sin(raan_rad),
        -np.sin(inc_rad) * np.cos(raan_rad),
        np.cos(inc_rad)
    ])

    # Position in ECEF (Constant)
    r_site_ecef = llh2ecef(lat_rad, lon_rad, 0.0)
    
    total_sec = (jd_end - jd_start) * 86400.0
    t_array = np.arange(0, total_sec, step_sec)
    jd_array = jd_start + t_array / 86400.0
    
    dot_products = []
    
    for jd in jd_array:
        # Convert to ECI
        r_site_eci, _ = eci2_ecef_or_inverse_wrapper(r_site_ecef, jd)
        dot = np.dot(r_site_eci, N_orbit)
        dot_products.append(dot)
        
    dot_products = np.array(dot_products)
    
    # Find sign changes
    windows = []
    for i in range(len(dot_products) - 1):
        if np.sign(dot_products[i]) != np.sign(dot_products[i+1]):
            # Crosses plane!
            jd_cross = jd_array[i] # Approximate
            
            # Determine Ascending or Descending
            # If rate of dot product is positive or negative
            rate = dot_products[i+1] - dot_products[i]
            
            # Azimuth calculation
            cos_phi = np.cos(lat_rad)
            if cos_phi == 0:
                azimuth_deg = 0.0 # Pole
            else:
                sin_psi = np.cos(inc_rad) / cos_phi
                if abs(sin_psi) <= 1.0:
                    psi_rad = np.arcsin(sin_psi)
                    azimuth_deg = np.degrees(psi_rad)
                else:
                    azimuth_deg = np.nan # No launching directly
            
            is_ascending = rate > 0
            # Standard definition: Ascending pass means crossing going North, or similar.
            # Here we just flag it. If rate is positive, it is moving into the positive hemisphere of the normal.
            
            windows.append({
                'jd': jd_cross,
                'rate': rate,
                'azimuth_approx_deg': azimuth_deg
            })
            
    return windows

def eci2_ecef_or_inverse_wrapper(recef, jd):
    """Temporary local wrapper to use ecef2eci avoiding import loop or correct usage"""
    from gnc_toolkit.utils.frame_conversion import ecef2eci
    return ecef2eci(recef, np.zeros(3), jd)

def compute_injection_state(lat_deg, lon_deg, alt_m, azimuth_deg, flight_path_angle_deg, speed_mps, jd):
    """
    Computes ECI state vector at insertion.
    
    Args:
        lat_deg (float): Latitude [deg].
        lon_deg (float): Longitude [deg].
        alt_m (float): Altitude [m].
        azimuth_deg (float): Azimuth from North [deg].
        flight_path_angle_deg (float): Flight Path Angle from horizontal [deg].
        speed_mps (float): Speed magnitude [m/s].
        jd (float): Julian Date at injection.
        
    Returns:
        tuple: (r_eci [m], v_eci [m/s])
    """
    from gnc_toolkit.utils.frame_conversion import ecef2eci
    
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    az_rad = np.radians(azimuth_deg)
    fpa_rad = np.radians(flight_path_angle_deg)
    
    # 1. Position in ECEF
    r_ecef = llh2ecef(lat_rad, lon_rad, alt_m)
    
    # 2. Velocity in Topocentric ENU
    # Speed components
    # V_North = Speed * cos(FPA) * cos(Az)
    # V_East = Speed * cos(FPA) * sin(Az)
    # V_Up = Speed * sin(FPA)
    v_east = speed_mps * np.cos(fpa_rad) * np.sin(az_rad)
    v_north = speed_mps * np.cos(fpa_rad) * np.cos(az_rad)
    v_up = speed_mps * np.sin(fpa_rad)
    
    v_enu = np.array([v_east, v_north, v_up])
    
    # 3. Rotation ENU to ECEF
    # ENU coords: E, N, U
    # E = [-sin_lon, cos_lon, 0]
    # N = [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat]
    # U = [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]
    cos_lat, sin_lat = np.cos(lat_rad), np.sin(lat_rad)
    cos_lon, sin_lon = np.cos(lon_rad), np.sin(lon_rad)
    
    E = np.array([-sin_lon, cos_lon, 0])
    N_vec = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    U = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    
    R_enu2ecef = np.vstack((E, N_vec, U)).T # Columns are E, N, U
    
    v_ecef_rel = R_enu2ecef @ v_enu
    
    # Add Coriolis (Earth rotation) to get inertial velocity represented in ECEF
    # V_inertial = V_rel + omega x r
    omega_earth = 7.292115e-5 # rad/s
    omega_vec = np.array([0, 0, omega_earth])
    
    v_ecef_inertial = v_ecef_rel + np.cross(omega_vec, r_ecef)
    
    # 4. Convert to ECI
    r_eci, v_eci = ecef2eci(r_ecef, v_ecef_inertial, jd)
    
    return r_eci, v_eci
