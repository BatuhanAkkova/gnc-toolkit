import pytest
import numpy as np
from gnc_toolkit.gravity import point_mass_grav, j2_grav

def test_point_mass_grav():
    r = np.array([7000000.0, 0, 0]) # 7000 km
    a = point_mass_grav(r)
    
    # a = -mu/r^2 * r_hat
    mu = 398600.4418e9
    expected_mag = mu / (7000000.0**2)
    
    assert np.isclose(np.linalg.norm(a), expected_mag, rtol=1e-5)
    # direction opposite to r
    assert np.allclose(a / np.linalg.norm(a), -r / np.linalg.norm(r))

def test_j2_grav():
    r = np.array([7000000.0, 0, 0])
    v = np.array([0, 7500.0, 0])
    dt = 10.0
    
    r_new, v_new = j2_grav(r, v, dt)
    
    assert r_new.shape == (3,)
    assert v_new.shape == (3,)
    
    # Check if moved
    assert not np.allclose(r_new, r)
