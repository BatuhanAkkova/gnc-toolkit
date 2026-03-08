import numpy as np
import pytest
from gnc_toolkit.navigation.iod import gibbs_iod, herrick_gibbs_iod
from gnc_toolkit.utils.state_to_elements import kepler2eci

def test_gibbs_iod():
    # Create synthetic data
    a = 7000e3
    ecc = 0.1
    incl = np.radians(28.5)
    raan = 0
    argp = 0
    
    # t1, t2, t3
    taus = [0, 100, 200]
    positions = []
    actual_v2 = None
    
    mu = 398600.4415e9
    n = np.sqrt(mu / a**3)
    
    for i, t in enumerate(taus):
        nu = n * t # Approx for small t, but let's be precise if needed
        r, v = kepler2eci(a, ecc, incl, raan, argp, nu)
        positions.append(r)
        if i == 1:
            actual_v2 = v
            
    v2_est = gibbs_iod(positions[0], positions[1], positions[2])
    
    # Gibbs usually needs larger separation, let's see
    assert v2_est == pytest.approx(actual_v2, rel=0.01)

def test_herrick_gibbs_iod():
    a = 7000e3
    ecc = 0.01
    incl = np.radians(28.5)
    raan = 0
    argp = 0
    
    dt = 10 
    taus = [0, dt, 2*dt]
    positions = []
    actual_v2 = None
    
    mu = 398600.4415e9
    n = np.sqrt(mu / a**3)
    
    for i, t in enumerate(taus):
        nu = n * t
        r, v = kepler2eci(a, ecc, incl, raan, argp, nu)
        positions.append(r)
        if i == 1:
            actual_v2 = v
            
    v2_est = herrick_gibbs_iod(positions[0], positions[1], positions[2], dt, dt)
    
    assert v2_est == pytest.approx(actual_v2, rel=1e-4)
