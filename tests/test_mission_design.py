import numpy as np
from gnc_toolkit.mission_design import (
    calculate_access_windows,
    calculate_ground_track,
    calculate_lighting_conditions,
    calculate_launch_windows,
    compute_injection_state,
    calculate_constellation_coverage,
    calculate_deployment_sequence
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

def test_calculate_constellation_coverage():
    t_array = np.array([0, 100, 200])
    
    # Target Ground Station at Equator, 0 Lon
    target_points = np.array([[0.0, 0.0, 0.0]])
    
    R_earth = 6378137.0
    from gnc_toolkit.utils.frame_conversion import llh2ecef, ecef2eci
    
    r_gs_ecef = llh2ecef(0, 0, 0)
    # Sat 1: Directly above
    r_sat1_ecef = r_gs_ecef * (R_earth + 500000.0) / R_earth
    r_sat1_eci, _ = ecef2eci(r_sat1_ecef, np.zeros(3), 2451545.0)
    
    # Sat 2: Behind Earth
    r_sat2_ecef = -r_gs_ecef * (R_earth + 500000.0) / R_earth
    r_sat2_eci, _ = ecef2eci(r_sat2_ecef, np.zeros(3), 2451545.0)
    
    # Sat 1 visible at t=0 and t=200, Sat 2 visible at t=100
    r_eci_array_1 = np.vstack((r_sat1_eci, r_sat2_eci, r_sat1_eci))
    
    # Second sat visible at t=100
    r_eci_array_2 = np.vstack((r_sat2_eci, r_sat1_eci, r_sat2_eci))
    
    r_eci_array_list = [r_eci_array_1, r_eci_array_2]
    
    res = calculate_constellation_coverage(t_array, r_eci_array_list, target_points, min_elevation_deg=5.0, jdut1=2451545.0)
    
    assert isinstance(res, list)
    assert len(res) == 1
    # Since either Sat 1 or Sat 2 is directly above the GS at any given time:
    # Coverage should be continuous, max gap should be 0, total coverage time should be 200
    assert res[0]['max_revisit_time_gap'] == 0.0
    assert res[0]['total_coverage_time'] > 0

def test_calculate_deployment_sequence():
    planes = 3
    sats_per_plane = 2
    phasing_f = 1
    # total sats = 6
    
    seq = calculate_deployment_sequence(planes, sats_per_plane, phasing_f, inc_deg=45.0)
    
    assert len(seq) == 6
    
    # Plane RAAN spacing = 360 / 3 = 120
    # In-plane spacing = 360 / 2 = 180
    # Between-plane TA phasing = 360 * 1 / 6 = 60
    
    # Sat 0: p=0, s=0 -> RAAN=0, TA=0
    assert seq[0]['raan_deg'] == 0.0
    assert seq[0]['ta_deg'] == 0.0
    
    # Sat 1: p=0, s=1 -> RAAN=0, TA=180
    assert seq[1]['raan_deg'] == 0.0
    assert seq[1]['ta_deg'] == 180.0
    
    # Sat 2: p=1, s=0 -> RAAN=120, TA=60
    assert seq[2]['raan_deg'] == 120.0
    assert seq[2]['ta_deg'] == 60.0


