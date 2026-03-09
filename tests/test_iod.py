import numpy as np
import pytest
from gnc_toolkit.navigation.iod import gibbs_iod, herrick_gibbs_iod, laplace_iod, laplace_iod_from_observations, gauss_iod
from gnc_toolkit.utils.state_to_elements import kepler2eci

def test_gauss_iod():
    # Synthetic orbit context
    a = 10000e3
    ecc = 0.05
    incl = np.radians(35.0)
    raan = np.radians(10.0)
    argp = np.radians(20.0)
    
    mu = 398600.4415e9
    n = np.sqrt(mu / a**3)
    
    # Separation for Gauss
    t2 = 1800.0
    dt = 300.0
    times = [t2 - dt, t2, t2 + dt]
    
    rho_hats = []
    Rs = []
    
    # Ground station / Observer (fixed in ECI for test)
    R_earth = 6378137.0
    lat = np.radians(30.0)
    R_obs = np.array([R_earth * np.cos(lat), 0, R_earth * np.sin(lat)])
    
    expected_state_t2 = None
    
    for t in times:
        nu = n * t
        r, v = kepler2eci(a, ecc, incl, raan, argp, nu)
        
        rho_vec = r - R_obs
        rho_hats.append(rho_vec / np.linalg.norm(rho_vec))
        Rs.append(R_obs)
        
        if abs(t - t2) < 1e-6:
            expected_state_t2 = np.concatenate([r, v])
            
    # Run Gauss IOD
    est_state = gauss_iod(
        rho_hats[0], rho_hats[1], rho_hats[2], 
        times[0], times[1], times[2], 
        Rs[0], Rs[1], Rs[2], mu
    )
    
    # Gauss is sensitive. For 10min total arc on 10k orbit, ~1% accuracy is typical for IOD
    assert est_state[0:3] == pytest.approx(expected_state_t2[0:3], rel=2e-2)
    assert est_state[3:6] == pytest.approx(expected_state_t2[3:6], rel=5e-2)

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

def test_laplace_iod_from_observations():
    a = 7500e3
    ecc = 0.0
    incl = np.radians(45.0)
    raan = np.radians(30.0)
    argp = np.radians(60.0)
    
    mu = 398600.4415e9
    n = np.sqrt(mu / a**3)
    
    t2 = 100.0
    dt = 10.0 # 10 seconds between observations
    times = [t2 - dt, t2, t2 + dt]
    
    rho_hats = []
    Rs = []
    
    # Observer position (fixed in ECI for simplicity)
    R_fixed = np.array([6378e3, 1000e3, 500e3])
    
    expected_state_t2 = None
    
    for t in times:
        nu = n * t
        r, v = kepler2eci(a, ecc, incl, raan, argp, nu)
        
        rho_vec = r - R_fixed
        rho_hats.append(rho_vec / np.linalg.norm(rho_vec))
        Rs.append(R_fixed)
        
        if t == t2:
            expected_state_t2 = np.concatenate([r, v])
            
    # Run Laplace IOD
    est_state = laplace_iod_from_observations(rho_hats, Rs, times, mu)
    
    # Check position and velocity
    # For a circular orbit and 10s step, accuracy should be high
    assert est_state[0:3] == pytest.approx(expected_state_t2[0:3], rel=1e-5)
    assert est_state[3:6] == pytest.approx(expected_state_t2[3:6], rel=1e-4)
