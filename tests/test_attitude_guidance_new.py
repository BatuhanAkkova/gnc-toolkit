import numpy as np
import pytest
from gnc_toolkit.guidance import (
    nadir_pointing_reference,
    sun_pointing_reference,
    target_tracking_reference,
    attitude_blending,
    eigenaxis_slew_path_planning
)
from gnc_toolkit.utils.quat_utils import quat_norm, quat_rot

def test_nadir_pointing():
    pos = np.array([7000e3, 0.0, 0.0]) # 7000 km on X
    vel = np.array([0.0, 7500.0, 0.0]) # 7.5 km/s on Y
    
    q = nadir_pointing_reference(pos, vel)
    
    assert quat_norm(q) == pytest.approx(1.0)
    
    # In Nadir frame: 
    # Body Z should align with -pos (towards Earth)
    # Body Y should align with -orb_normal
    # orb_normal = pos x vel = [0, 0, 7000e3 * 7500] (Positive Z)
    # So Body Y should align with -Z_eci
    
    body_z = quat_rot(q, [0, 0, 1.0])
    assert body_z[0] == pytest.approx(-1.0) # Aligned with -X_eci
    
    body_y = quat_rot(q, [0, 1.0, 0.0])
    assert body_y[2] == pytest.approx(-1.0) # Aligned with -Z_eci

def test_sun_pointing():
    sun_vec = np.array([0.0, 1.0, 0.0]) # Sun on Y_eci
    # Primary axis is Body X
    q = sun_pointing_reference(sun_vec)
    
    body_x = quat_rot(q, [1.0, 0.0, 0.0])
    assert body_x[0] == pytest.approx(0.0, abs=1e-7)
    assert body_x[1] == pytest.approx(1.0)

def test_attitude_blending():
    q1 = np.array([0, 0, 0, 1.0]) # Identity
    # 90 deg rotation about Z
    q2 = np.array([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
    
    # Halfway (45 deg)
    q_mid = attitude_blending(q1, q2, 0.5)
    
    expected_mid = np.array([0, 0, np.sin(np.pi/8), np.cos(np.pi/8)])
    np.testing.assert_allclose(q_mid, expected_mid, atol=1e-7)

def test_target_tracking():
    pos = np.array([0, 0, 0])
    target = np.array([100.0, 100.0, 100.0])
    
    q = target_tracking_reference(pos, target)
    
    # Body Z should point towards target
    body_z = quat_rot(q, [0, 0, 1.0])
    expected_z = target / np.linalg.norm(target)
    np.testing.assert_allclose(body_z, expected_z, atol=1e-7)
