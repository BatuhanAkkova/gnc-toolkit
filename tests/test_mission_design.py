import numpy as np
from gnc_toolkit.mission_design import (
    calculate_access_windows,
    calculate_ground_track,
    calculate_lighting_conditions,
    calculate_launch_windows,
    compute_injection_state
)

def test_calculate_access_windows():
    # Setup test or simulation
    t_array = np.array([0, 100, 200])
    
    # Ground Station at Equator, 0 Lon
    gs_lat = 0.0
    gs_lon = 0.0
    gs_alt = 0.0
    
    # Sat positions
    # 1. Directly above at 500km altitude
    # 2. Behind Earth
    # 3. Off to the side
    
    R_earth = 6378137.0
    # at t=0, GMST might not be 0. Let's use jdut1=2451545.0 (GMST ~ 280 deg)
    # To make it easy, place the sat directly above the GS location in ECEF, then use ecef2eci to get r_eci
    from gnc_toolkit.utils.frame_conversion import llh2ecef, ecef2eci
    
    r_gs_ecef = llh2ecef(0, 0, 0) # [R_earth, 0, 0]
    r_sat_ecef_above = r_gs_ecef * (R_earth + 500000.0) / R_earth
    r_sat_eci_above, _ = ecef2eci(r_sat_ecef_above, np.zeros(3), 2451545.0)
    
    r_sat_ecef_below = -r_gs_ecef * (R_earth + 500000.0) / R_earth
    r_sat_eci_below, _ = ecef2eci(r_sat_ecef_below, np.zeros(3), 2451545.0)
    
    r_eci_array = np.vstack((r_sat_eci_above, r_sat_eci_below, r_sat_eci_above))
    
    res = calculate_access_windows(t_array, r_eci_array, gs_lat, gs_lon, gs_alt, min_elevation_deg=5.0, jdut1=2451545.0)
    
    assert 'visible_intervals' in res
    assert 'elevation_history' in res
    assert len(res['visible_intervals']) == 2 # 0 and 200 visible
    assert res['elevation_history'][0] > 80.0 # Directly above
    assert res['elevation_history'][1] < 0.0 # Below horizon

def test_calculate_ground_track():
    t_array = np.array([0])
    R_earth = 6378137.0
    r_eci = np.array([R_earth + 500000.0, 0, 0])
    
    res = calculate_ground_track(t_array, np.array([r_eci]), jdut1=2451545.0)
    
    assert 'lat_deg' in res
    assert 'lon_deg' in res
    assert 'alt_m' in res
    assert len(res['lat_deg']) == 1

def test_calculate_lighting_conditions():
    t_array = np.array([0])
    R_earth = 6378137.0
    r_eci = np.array([0, R_earth + 500000.0, 0])
    v_eci = np.array([1, 0, 0]) # Just some velocity
    
    res = calculate_lighting_conditions(t_array, np.array([r_eci]), np.array([v_eci]), jdut1=2451545.0)
    
    assert 'beta_angle_deg' in res
    assert 'eclipse_state' in res
    assert len(res['beta_angle_deg']) == 1

def test_calculate_launch_windows():
    jd_start = 2451545.0
    jd_end = 2451546.0 # 1 day
    inc = 45.0
    raan = 0.0
    lat = 28.5 # Cape Canaveral
    lon = -80.5
    
    res = calculate_launch_windows(jd_start, jd_end, inc, raan, lat, lon)
    
    assert isinstance(res, list)
    # Usually 2 passes per day
    assert len(res) >= 1

def test_compute_injection_state():
    lat = 28.5
    lon = -80.5
    alt = 200000.0
    azimuth = 90.0 # Eastward
    fpa = 0.0 # Horizontal
    speed = 7800.0 # Orbital speed m/s
    jd = 2451545.0
    
    r_eci, v_eci = compute_injection_state(lat, lon, alt, azimuth, fpa, speed, jd)
    
    assert len(r_eci) == 3
    assert len(v_eci) == 3
    # Check altitude
    assert abs(np.linalg.norm(r_eci) - (6378137.0 + alt)) < 20000.0 # Geodetic to Cartesian distance deviation
