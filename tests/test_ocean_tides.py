import pytest
import numpy as np
from gnc_toolkit.disturbances.gravity import OceanTidesGravity

@pytest.fixture
def state():
    # Position: ~7000 km altitude
    r_eci = np.array([7000e3, 0.0, 0.0]) # meters
    jd = 2460337.5 # Arbitrary date
    return r_eci, jd

def test_ocean_tides_acceleration_non_zero(state):
    r_eci, jd = state
    model = OceanTidesGravity()
    acc = model.get_acceleration(r_eci, jd)
    
    # Acceleration should be non-zero
    assert np.any(acc != 0.0)
    
    # Magnitude should be small (typical ocean tide acceleration is < 1e-7 m/s^2)
    mag = np.linalg.norm(acc)
    assert mag < 1e-6
    assert mag > 1e-12 # Should be at least something

def test_ocean_tides_time_varying(state):
    r_eci, jd = state
    model = OceanTidesGravity()
    acc1 = model.get_acceleration(r_eci, jd)
    acc2 = model.get_acceleration(r_eci, jd + 0.5) # 12 hours later
    
    # Acceleration should change with time
    assert not np.allclose(acc1, acc2, atol=1e-15, rtol=1e-12)

def test_ocean_tides_altitude_dependence():
    model = OceanTidesGravity()
    # Close orbit
    acc_close = model.get_acceleration(np.array([7000e3, 0, 0]), 2460337.5)
    # Far orbit
    acc_far = model.get_acceleration(np.array([42000e3, 0, 0]), 2460337.5)
    
    mag_close = np.linalg.norm(acc_close)
    mag_far = np.linalg.norm(acc_far)
    
    # Gravity falls off with distance. Ocean tides fall off with (Re/r)^(n+1) = (Re/r)^3 for n=2.
    # So far orbit should have much smaller acceleration.
    assert mag_far < mag_close
