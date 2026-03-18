import numpy as np
from gnc_toolkit.utils.frame_conversion import eci2ecef, eci2llh, llh2ecef
from gnc_toolkit.environment.solar import Sun

def calculate_access_windows(t_array, r_eci_array, gs_lat_deg, gs_lon_deg, gs_alt_m, min_elevation_deg=5.0, jdut1=2451545.0):
    """
    Calculates access windows (visibility) from a Ground Station for a given trajectory.
    
    Args:
        t_array (np.ndarray): Time array [s] from starting epoch.
        r_eci_array (np.ndarray): Position history in ECI [m], shape (N, 3).
        gs_lat_deg (float): Ground Station Latitude [deg].
        gs_lon_deg (float): Ground Station Longitude [deg].
        gs_alt_m (float): Ground Station Altitude [m].
        min_elevation_deg (float): Minimum elevation for visibility [deg].
        jdut1 (float): Julian Date UT1 at t=0.
        
    Returns:
        dict: Containing 'visible_intervals' (list of dicts), 'elevation_history' (degrees).
    """
    gs_lat_rad = np.radians(gs_lat_deg)
    gs_lon_rad = np.radians(gs_lon_deg)
    r_gs_ecef = llh2ecef(gs_lat_rad, gs_lon_rad, gs_alt_m)

    # Zenith vector at GS (Normal to ellipsoid)
    # n = [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]
    U = np.array([
        np.cos(gs_lat_rad) * np.cos(gs_lon_rad),
        np.cos(gs_lat_rad) * np.sin(gs_lon_rad),
        np.sin(gs_lat_rad)
    ])

    elevations = []
    visibility = []
    
    for i, t in enumerate(t_array):
        jd_current = jdut1 + t / 86400.0
        r_eci = r_eci_array[i]
        
        # Convert to ECEF
        # eci2ecef expects reci, veci. We only need position here.
        r_ecef, _ = eci2ecef(r_eci, np.zeros(3), jd_current)
        
        # Relative vector
        rho_ecef = r_ecef - r_gs_ecef
        rho_norm = np.linalg.norm(rho_ecef)
        
        if rho_norm == 0:
            elevations.append(90.0)
            visibility.append(True)
            continue
            
        # Elevation
        elevation_rad = np.arcsin(np.dot(rho_ecef, U) / rho_norm)
        elevation_deg = np.degrees(elevation_rad)
        
        elevations.append(elevation_deg)
        visibility.append(elevation_deg >= min_elevation_deg)

    # Find intervals
    visible_intervals = []
    in_pass = False
    start_time = None
    
    for i, is_visible in enumerate(visibility):
        if is_visible and not in_pass:
            in_pass = True
            start_time = t_array[i]
        elif not is_visible and in_pass:
            in_pass = False
            visible_intervals.append({
                'start_time': start_time,
                'end_time': t_array[i],
                'duration': t_array[i] - start_time
            })
            
    if in_pass:
        visible_intervals.append({
            'start_time': start_time,
            'end_time': t_array[-1],
            'duration': t_array[-1] - start_time
        })

    return {
        'visible_intervals': visible_intervals,
        'elevation_history': np.array(elevations)
    }

def calculate_ground_track(t_array, r_eci_array, jdut1=2451545.0):
    """
    Calculates ground track coordinates (Lat, Lon, Alt) over time.
    
    Args:
        t_array (np.ndarray): Time array [s] from starting epoch.
        r_eci_array (np.ndarray): Position history in ECI [m], shape (N, 3).
        jdut1 (float): Julian Date UT1 at t=0.
        
    Returns:
        dict: Containing 'lat_deg', 'lon_deg', 'alt_m' arrays.
    """
    lats = []
    lons = []
    alts = []
    
    for i, t in enumerate(t_array):
        jd_current = jdut1 + t / 86400.0
        r_eci = r_eci_array[i]
        
        lat_rad, lon_rad, alt = eci2llh(r_eci, jd_current)
        
        # Normalize lon to [-180, 180]
        lon_deg = np.degrees(lon_rad)
        if lon_deg > 180:
            lon_deg -= 360
        elif lon_deg < -180:
            lon_deg += 360
            
        lats.append(np.degrees(lat_rad))
        lons.append(lon_deg)
        alts.append(alt)
        
    return {
        'lat_deg': np.array(lats),
        'lon_deg': np.array(lons),
        'alt_m': np.array(alts)
    }

def calculate_lighting_conditions(t_array, r_eci_array, v_eci_array, jdut1=2451545.0):
    """
    Calculates Beta angle and Eclipse status over time.
    
    Args:
        t_array (np.ndarray): Time array [s] from starting epoch.
        r_eci_array (np.ndarray): Position history in ECI [m], shape (N, 3).
        v_eci_array (np.ndarray): Velocity history in ECI [m/s], shape (N, 3).
        jdut1 (float): Julian Date UT1 at t=0.
        
    Returns:
        dict: Containing 'beta_angle_deg', 'eclipse_state' (1 for Sun, 0 for shade).
    """
    sun_model = Sun()
    
    beta_angles = []
    eclipse_state = []
    
    R_earth = 6378137.0
    
    for i, t in enumerate(t_array):
        jd_current = jdut1 + t / 86400.0
        r_eci = r_eci_array[i]
        v_eci = v_eci_array[i]
        
        # Sun position
        r_sun = sun_model.calculate_sun_eci(jd_current)
        r_sun_norm = np.linalg.norm(r_sun)
        u_sun = r_sun / r_sun_norm
        
        # Orbit Normal
        h = np.cross(r_eci, v_eci)
        h_norm = np.linalg.norm(h)
        if h_norm == 0:
            beta_angles.append(0.0)
        else:
            u_h = h / h_norm
            # Beta angle = asin(dot(h, u_sun))
            # Wait, beta angle is the angle between Sun and Orbit Plane.
            # Normal is orthogonal to the plane.
            # Angle to plane = 90 - angle to normal.
            # dot(u_h, u_sun) = cos(angle to normal).
            # sin(angle to plane) = cos(angle to normal).
            # So beta = asin(dot(u_h, u_sun)). Correct.
            beta_rad = np.arcsin(np.dot(u_h, u_sun))
            beta_angles.append(np.degrees(beta_rad))
            
        # Eclipse Check (Cylindrical model)
        # Projection of r_eci onto Sun vector
        s = np.dot(r_eci, u_sun)
        
        if s > 0:
            eclipse_state.append(1.0) # Sunlight
        else:
            # Perpendicular distance
            r_perp_sq = np.dot(r_eci, r_eci) - s*s
            if r_perp_sq < R_earth**2:
                eclipse_state.append(0.0) # Eclipse
            else:
                eclipse_state.append(1.0) # Sunlight
                
    return {
        'beta_angle_deg': np.array(beta_angles),
        'eclipse_state': np.array(eclipse_state)
    }
