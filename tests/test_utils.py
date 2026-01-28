import pytest
import numpy as np
from src.utils.quat_utils import (
    quat_normalize, quat_conj, quat_norm, quat_inv,
    quat_mult, quat_rot, quat_to_rmat, axis_angle_to_quat
)
from src.utils.time_utils import (
    calc_jd, jd_to_datetime, day_to_mdtime, calc_gmst,
    calc_last, calc_lst, calc_doy, is_leap_year, convert_time
)
from src.utils.state_to_elements import eci2kepler, kepler2eci
from src.utils.frame_conversion import eci2ecef, ecef2eci, eci2lvlh_dcm, eci2llh, elements2perifocal_dcm
from src.utils.state_conversion import (
    quat_to_dcm, quat_to_euler, dcm_to_quat, dcm_to_euler,
    euler_to_quat, euler_to_dcm, rot_x, rot_y, rot_z
)

# QUAT UTILS

def test_quat_norm_normalize():
    q = np.array([3, 0, 4, 0]) # Norm is 5
    assert np.isclose(quat_norm(q), 5.0)
    
    q_norm = quat_normalize(q)
    assert np.isclose(np.linalg.norm(q_norm), 1.0)
    assert np.allclose(q_norm, np.array([0.6, 0, 0.8, 0]))

    with pytest.raises(ValueError):
        quat_normalize(np.zeros(4))

def test_quat_ops_basic():
    q = np.array([0, 0, 0, 1]) # Identity
    
    # Conjugate
    q_conj = quat_conj(q)
    np.testing.assert_array_equal(q_conj, np.array([0, 0, 0, 1]))
    
    # Inverse
    q_inv = quat_inv(q)
    np.testing.assert_array_equal(q_inv, np.array([0, 0, 0, 1]))
    
    q2 = np.array([1, 0, 0, 0])
    q_mult = quat_mult(q2, q2)
    np.testing.assert_array_equal(q_mult, np.array([0, 0, 0, -1]))

def test_quat_rot():
    # Rotate vector [1, 0, 0] by 90 deg around z-axis
    # q = [0, 0, sin(45), cos(45)] = [0, 0, 0.7071, 0.7071]
    angle = np.pi/2
    q = np.array([0, 0, np.sin(angle/2), np.cos(angle/2)])
    
    v = np.array([1, 0, 0])
    v_rot = quat_rot(q, v)
    
    # Expect [0, 1, 0]
    np.testing.assert_allclose(v_rot, np.array([0, 1, 0]), atol=1e-10)

def test_axis_angle_to_quat():
    axis = np.array([0, 0, 1], dtype=float)
    angle = np.pi/2
    axis_angle = axis * angle
    
    q = axis_angle_to_quat(axis_angle)
    expected = np.array([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
    
    np.testing.assert_allclose(q, expected, atol=1e-10)
    
    # Zero rotation
    q_zero = axis_angle_to_quat(np.zeros(3))
    np.testing.assert_array_equal(q_zero, np.array([0, 0, 0, 1]))

def test_quat_to_rmat():
    # Identity
    q = np.array([0, 0, 0, 1])
    rmat = quat_to_rmat(q)
    np.testing.assert_array_equal(rmat, np.eye(3))
    
    # 90 deg Z rotation
    # q = [0, 0, 1/sqrt(2), 1/sqrt(2)]
    val = 1.0/np.sqrt(2)
    q_z = np.array([0, 0, val, val])
    rmat_z = quat_to_rmat(q_z)
    
    expected = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    np.testing.assert_allclose(rmat_z, expected, atol=1e-10)

# TIME UTILS

def test_calc_jd():
    # J2000 Epoch: 2000 Jan 1 12:00:00 TT -> JD 2451545.0
    jd, jdfrac = calc_jd(2000, 1, 1, 12, 0, 0)
    assert jd + jdfrac == 2451545.0
    
def test_jd_datetime_roundtrip():
    y, m, d, h, mn, s = 2024, 1, 18, 10, 30, 45.0
    jd, jdfrac = calc_jd(y, m, d, h, mn, s)
    y2, m2, d2, h2, mn2, s2 = jd_to_datetime(jd, jdfrac)
    
    assert y == y2
    assert m == m2
    assert d == d2
    assert h == h2
    assert mn == mn2
    assert np.isclose(s, s2, atol=1e-6)

def test_calc_gmst():
    # J2000 GMST verify
    jd = 2451545.0
    gmst_rad = calc_gmst(jd)
    gmst_deg = np.degrees(gmst_rad)
    
    # Expected approx 280.4606 deg
    assert np.isclose(gmst_deg, 280.4606, atol=1e-2)
    
    # Range check
    assert 0 <= gmst_rad < 2*np.pi

def test_is_leap_year():
    assert is_leap_year(2000)
    assert is_leap_year(2024)
    assert not is_leap_year(2100)
    assert not is_leap_year(2023)

def test_calc_doy():
    assert calc_doy(2024, 1, 1) == 1
    assert calc_doy(2024, 2, 29) == 60
    assert calc_doy(2023, 2, 28) == 59
    assert calc_doy(2023, 3, 1) == 60

def test_calc_lst():
    # LST = GMST + lon
    gmst = 1.0
    lon = 0.5
    lst = calc_lst(gmst, lon)
    assert np.isclose(lst, 1.5)

def test_convert_time():
    # Should run without error
    res = convert_time(2024, 1, 1, 12, 0, 0, 0, 0, 0, 37)
    assert isinstance(res, tuple)
    assert len(res) == 15

# STATE TRANSFORM

def test_state_transform_roundtrip():
    # Define a state in ECI (Meters, m/s)
    # 7000 km orbit, circular-ish
    r = np.array([7000000.0, 0, 0])
    mu = 398600.4415e9
    v_mag = np.sqrt(mu / np.linalg.norm(r))
    v = np.array([0, v_mag, 0])
    
    a, ecc, incl, raan, argp, nu, M, E, p, arglat, truelon, lonper = eci2kepler(r, v)
    
    assert np.isclose(a, 7000000.0, rtol=1e-6)
    assert np.isclose(ecc, 0, rtol=1e-6)
    assert np.isclose(incl, 0, rtol=1e-6)
    assert np.isclose(raan, 0, rtol=1e-6)
    assert np.isclose(argp, 0, rtol=1e-6)
    assert np.isclose(nu, 0, rtol=1e-6)
    assert np.isclose(M, 0, rtol=1e-6)
    assert np.isclose(E, 0, rtol=1e-6)
    assert np.isclose(p, 7000000.0, rtol=1e-6)
    assert np.isclose(arglat, 0, rtol=1e-6)
    assert np.isclose(truelon, 0, rtol=1e-6)
    assert np.isclose(lonper, 0, rtol=1e-6)
    
    # Round trip
    r_back, v_back = kepler2eci(a, ecc, incl, raan, argp, nu)
    
    np.testing.assert_allclose(r_back, r, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(v_back, v, rtol=1e-6, atol=1e-6)

# CONVERSIONS

def test_eci_ecef_roundtrip():
    """Test ECI to ECEF and back."""
    reci = np.array([7000, 0, 0])
    veci = np.array([0, 7.5, 0])
    jdut1 = 2451545.0 # Some Julian Date
    
    recef, vecef = eci2ecef(reci, veci, jdut1)
    reci_back, veci_back = ecef2eci(recef, vecef, jdut1)
    
    np.testing.assert_allclose(reci, reci_back, atol=1e-10)
    np.testing.assert_allclose(veci, veci_back, atol=1e-10)

def test_eci2lvlh_dcm():
    """Test ECI to LVLH direction cosine matrix."""
    r = np.array([7000, 0, 0])
    v = np.array([0, 7.5, 0])
    
    # Orbit normal h = r x v = (0, 0, 7000*7.5) = +Z direction
    # y_lvlh = -h/norm = (0, 0, -1)
    # z_lvlh = -r/norm = (-1, 0, 0)
    # x_lvlh = y x z = (0, 0, -1) x (-1, 0, 0) = (0, 1, 0)
    
    expected_dcm = np.array([
        [0, 1, 0],
        [0, 0, -1],
        [-1, 0, 0]
    ])
    
    dcm = eci2lvlh_dcm(r, v)
    np.testing.assert_allclose(dcm, expected_dcm, atol=1e-10)

def test_eci2llh():
    r_test = [6678137.0, 0, 0]
    jdut1 = 2451545.0
    lat, lon, h = eci2llh(r_test, jdut1)
    np.testing.assert_allclose(h, 300000, atol=1e-10)

def test_elements2perifocal_dcm():
    """Test elements to perifocal direction cosine matrix."""
    # Case: Identity (all zero)
    dcm = elements2perifocal_dcm(0, 0, 0)
    np.testing.assert_allclose(dcm, np.eye(3), atol=1e-10)
    
    # Case: 90 deg inclination (x-rotation)
    dcm_inc = elements2perifocal_dcm(0, np.pi/2, 0)
    np.testing.assert_allclose(np.dot(dcm_inc, dcm_inc.T), np.eye(3), atol=1e-10)
    
    # Explicit check for 90 deg inclination (Rotation about X axis of perifocal? No, ECI X if node is there)
    # R = [1, 0, 0; 0, 0, -1; 0, 1, 0]
    expected_inc = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    np.testing.assert_allclose(dcm_inc, expected_inc, atol=1e-10)

    # Case: RAAN 90 (z-rotation)
    # R = Rz(90) = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    dcm_raan = elements2perifocal_dcm(np.pi/2, 0, 0)
    expected_raan = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    np.testing.assert_allclose(dcm_raan, expected_raan, atol=1e-10)

def test_rotation_matrices():
    """Test rotation matrices rot_x, rot_y, rot_z."""
    angle = np.pi / 2
    
    Rx = rot_x(angle)
    expected_Rx = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    np.testing.assert_allclose(Rx, expected_Rx, atol=1e-10)
    
    Ry = rot_y(angle)
    expected_Ry = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    np.testing.assert_allclose(Ry, expected_Ry, atol=1e-10)
    
    Rz = rot_z(angle)
    expected_Rz = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    np.testing.assert_allclose(Rz, expected_Rz, atol=1e-10)

def test_quat_dcm_roundtrip():
    """Test quaternion to direction cosine matrix and back."""
    # Identity
    q_id = np.array([0, 0, 0, 1])
    dcm_id = quat_to_dcm(q_id)
    np.testing.assert_allclose(dcm_id, np.eye(3), atol=1e-10)
    q_out = dcm_to_quat(dcm_id)
    # Quaternion double cover check
    if np.dot(q_id, q_out) < 0:
        q_out = -q_out
    np.testing.assert_allclose(q_out, q_id, atol=1e-10)

    # Random rotation
    angle = np.pi / 3
    axis = np.array([1, 1, 1])
    axis = axis / np.linalg.norm(axis)
    q_rot = np.array([
        axis[0] * np.sin(angle/2),
        axis[1] * np.sin(angle/2),
        axis[2] * np.sin(angle/2),
        np.cos(angle/2)
    ])
    
    dcm = quat_to_dcm(q_rot)
    q_res = dcm_to_quat(dcm)
    
    if np.dot(q_rot, q_res) < 0:
        q_res = -q_res
    np.testing.assert_allclose(q_res, q_rot, atol=1e-10)

# FRAME CONVERSIONS

def test_euler_dcm_roundtrip():
    """Test Euler angles to direction cosine matrix and back."""
    # Sequence 3-2-1 (scipy style z-y-x)
    angles = np.array([0.1, 0.2, 0.3])
    seq = "321"
    
    dcm = euler_to_dcm(angles, seq)
    angles_out = dcm_to_euler(dcm, seq)
    
    np.testing.assert_allclose(angles_out, angles, atol=1e-10)

def test_euler_quat_roundtrip():
    """Test Euler angles to quaternion and back."""
    angles = np.array([0.5, -0.2, 0.1])
    seq = "123"
    
    q = euler_to_quat(angles, seq)
    dcm_from_q = quat_to_dcm(q)
    dcm_from_e = euler_to_dcm(angles, seq)
    
    np.testing.assert_allclose(dcm_from_q, dcm_from_e, atol=1e-10)
    
    angles_back = quat_to_euler(q, seq)
    np.testing.assert_allclose(angles_back, angles, atol=1e-10)

def test_invalid_sequence():
    """Test invalid Euler angle sequences."""
    with pytest.raises(ValueError):
        euler_to_dcm([0,0,0], "12")
    with pytest.raises(ValueError):
        euler_to_quat([0,0,0], "1234")