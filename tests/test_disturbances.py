import pytest
import numpy as np
import os
from gnc_toolkit.disturbances.gravity import TwoBodyGravity, J2Gravity, HarmonicsGravity
from gnc_toolkit.disturbances.drag import LumpedDrag
from gnc_toolkit.disturbances.srp import Canonball
from gnc_toolkit.environment.density import Exponential

# Constants
MU = 398600.4418e9 # m^3/s^2

@pytest.fixture
def ephemeris():
    r_eci = np.array([7000e3, 0.0, 0.0]) # 7000 km altitude on X-axis
    v_eci = np.array([0.0, 7.5e3, 0.0]) # Circular velocity
    jd = 2460337.5 # Arbitrary date
    return r_eci, v_eci, jd

def test_two_body_gravity(ephemeris):
    r_eci, _, _ = ephemeris
    model = TwoBodyGravity(mu=MU)
    acc = model.get_acceleration(r_eci)
    
    r_norm = np.linalg.norm(r_eci)
    expected_acc = -MU / r_norm**3 * r_eci
    
    np.testing.assert_allclose(acc, expected_acc, rtol=1e-5)

def test_j2_gravity_equator(ephemeris):
    # At equator (z=0), J2 perturbs only in radial direction (no z-component)
    r_eci, _, _ = ephemeris
    model = J2Gravity(mu=MU, j2=0.001082635855)
    acc = model.get_acceleration(r_eci)
    
    # Check z-component is zero
    assert abs(acc[2]) < 1e-9

def test_j2_gravity_polar():
    # At pole, check for non-zero J2 effect
    r_eci = np.array([0.0, 0.0, 7000e3])
    model = J2Gravity(mu=MU, j2=0.001082635855)
    acc = model.get_acceleration(r_eci)
    
    two_body = TwoBodyGravity(mu=MU).get_acceleration(r_eci)
    
    # J2 should add perturbation
    assert not np.allclose(acc, two_body)

def test_harmonics_gravity_loading():
    # Test if it can load the file using default logic
    model = HarmonicsGravity(mu=MU, n_max=2, m_max=2)
    # Check if coefficients were loaded (sum of abs values should be non-zero)
    assert np.any(model.C != 0) or np.any(model.S != 0)

def test_drag_opposes_velocity(ephemeris):
    r_eci, v_eci, jd = ephemeris
    density_model = Exponential(rho0=1e-12) # Fake density model
    model = LumpedDrag(density_model)
    
    mass = 100.0
    area = 1.0
    cd = 2.2
    
    acc = model.get_acceleration(r_eci, v_eci, jd, mass, area, cd)
    
    # Drag should generally oppose velocity
    # Check dot product is negative
    assert np.dot(acc, v_eci) < 0

def test_srp_direction(ephemeris):
    r_eci, _, jd = ephemeris
    model = Canonball()
    
    mass = 100.0
    area = 1.0
    cr = 1.0
    
    acc = model.get_acceleration(r_eci, jd, mass, area, cr)

    # This test is weak without mocking sun position, but checks basic execution
    assert acc.shape == (3,)
