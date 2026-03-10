import numpy as np
import pytest
from gnc_toolkit.disturbances.gravity import ThirdBodyGravity, RelativisticCorrection
from gnc_toolkit.utils.time_utils import calc_jd

def test_third_body_gravity():
    model = ThirdBodyGravity()
    # Position in ECI [m]
    r_eci = np.array([7000.0, 0.0, 0.0]) * 1000.0
    jd = 2451545.0 + 0.5 # J2000 noon
    
    acc = model.get_acceleration(r_eci, jd)
    assert np.linalg.norm(acc) > 0
    # Third body (Sun/Moon) accelerations are small, order of 1e-6 to 1e-7 m/s^2 for LEO
    assert np.linalg.norm(acc) < 1e-5

def test_relativistic_correction():
    model = RelativisticCorrection()
    r_eci = np.array([7000.0, 0.0, 0.0]) * 1000.0
    v_eci = np.array([0.0, 7.5, 0.0]) * 1000.0 # 7.5 km/s
    
    acc = model.get_acceleration(r_eci, v_eci)
    assert np.linalg.norm(acc) > 0
    # Relativistic corrections are very small, order of 1e-9 m/s^2
    assert np.linalg.norm(acc) < 1e-8
