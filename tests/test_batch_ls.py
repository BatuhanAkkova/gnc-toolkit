import numpy as np
import pytest
from gnc_toolkit.navigation.batch_ls import BatchLeastSquaresOD
from gnc_toolkit.utils.state_to_elements import kepler2eci

def test_batch_ls_od():
    # True state
    a = 7000e3
    ecc = 0.001
    incl = np.radians(45)
    raan = np.radians(30)
    argp = np.radians(10)
    nu = 0
    
    x_true, _ = kepler2eci(a, ecc, incl, raan, argp, nu)
    # Full state
    v_true = np.array([0, 7500, 0]) # Simple circular-ish velocity
    # We need a consistent v_true for a=7000e3
    _, v_true = kepler2eci(a, ecc, incl, raan, argp, nu)
    x0_true = np.concatenate([x_true, v_true])
    
    # Generate observations
    times = [0, 10, 20, 30, 40, 50]
    observations = []
    
    # Simple propagation to generate synthetic observations
    # For testing BLS, we just need vectors that follow the model
    mu = 398600.4415e9
    n = np.sqrt(mu / a**3)
    
    for t in times:
        r_t, _ = kepler2eci(a, ecc, incl, raan, argp, nu + n*t)
        # Add some noise
        # observations.append(r_t + np.random.normal(0, 1.0, 3))
        observations.append(r_t) # No noise for first test
        
    # Initial guess with some error
    x_guess = x0_true + np.array([1000, -1000, 500, 1, -1, 0.5])
    
    bls = BatchLeastSquaresOD(x_guess)
    x_est = bls.solve(observations, times)
    
    assert x_est[:3] == pytest.approx(x0_true[:3], rel=1e-3)
    assert x_est[3:] == pytest.approx(x0_true[3:], rel=1e-3)
