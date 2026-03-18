import pytest
import numpy as np
from gnc_toolkit.propagators.sgp4_propagator import Sgp4Propagator

def test_sgp4_init_and_propagate():
    """
    Test SGP4 propagator with a sample ISS TLE.
    Verifies it propagates forward and returns valid dimensions and magnitudes.
    """
    # ISS TLE (Sample)
    line1 = "1 25544U 98067A   20325.43545139  .00001564  00000-0  36550-4 0  9997"
    line2 = "2 25544  51.6441 332.1863 0001461 319.4678  26.1557 15.49163286256157"
    
    sgp4_prop = Sgp4Propagator(line1, line2)
    
    # Propagate 1 orbit forward (~90 minutes = 5400 seconds)
    dt = 5400.0
    r_f, v_f = sgp4_prop.propagate(None, None, dt)
    
    assert len(r_f) == 3
    assert len(v_f) == 3
    
    # Check altitude magnitude is reasonable for LEO (approx 400 km)
    r_mag = np.linalg.norm(r_f)
    print(f"SGP4 R_mag: {r_mag/1000} km")
    assert 6700e3 < r_mag < 6900e3 # Radius of Earth + 300-500km
    
    v_mag = np.linalg.norm(v_f)
    print(f"SGP4 V_mag: {v_mag/1000} km/s")
    assert 7000 < v_mag < 8000 # Circular Velocity circa 7.7 km/s

def test_sgp4_propagate_to_jd():
    line1 = "1 25544U 98067A   20325.43545139  .00001564  00000-0  36550-4 0  9997"
    line2 = "2 25544  51.6441 332.1863 0001461 319.4678  26.1557 15.49163286256157"
    sgp4_prop = Sgp4Propagator(line1, line2)
    
    # Propagate to arbitrary JD
    jd_f = sgp4_prop.jdsatepoch + 0.1 # 0.1 days forward
    r_f, v_f = sgp4_prop.propagate_to_jd(jd_f, 0.0)
    
    assert len(r_f) == 3
    assert len(v_f) == 3
