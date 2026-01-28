import pytest
import numpy as np
from gnc_toolkit.two_body import two_body_kepler

def test_two_body_circular_period():
    # Circular orbit propagation by one period should return to same state
    
    mu_m = 398600.4418e9
    r_mag = 7000e3 # 7000 km in meters
    v_circ = np.sqrt(mu_m / r_mag)
    
    r0 = np.array([r_mag, 0, 0])
    v0 = np.array([0, v_circ, 0])
    
    period = 2 * np.pi * np.sqrt(r_mag**3 / mu_m)
    
    r_f, v_f = two_body_kepler(r0, v0, period)

    assert np.allclose(r_f, r0)
    assert np.allclose(v_f, v0)
