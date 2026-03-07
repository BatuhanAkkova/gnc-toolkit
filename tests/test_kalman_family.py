import pytest
import numpy as np
from gnc_toolkit.kalman_filters import SRUKF, EnKF, CKF, ParticleFilter, AKF, IMM, KF

def linear_fx(x, dt):
    # Constant velocity model
    F = np.array([[1, dt], [0, 1]])
    return np.dot(F, x)

def linear_hx(x):
    # Position measurement
    H = np.array([[1, 0]])
    return np.dot(H, x)

@pytest.fixture
def linear_setup():
    dt = 0.1
    dim_x = 2
    dim_z = 1
    x0 = np.array([0.0, 1.0])
    P0 = np.eye(2) * 0.1
    Q = np.eye(2) * 0.01
    R = np.array([[0.05]])
    return dt, dim_x, dim_z, x0, P0, Q, R

def test_sr_ukf(linear_setup):
    dt, dim_x, dim_z, x0, P0, Q, R = linear_setup
    filter = SRUKF(dim_x, dim_z)
    filter.x = x0.copy()
    filter.S = np.linalg.cholesky(P0)
    filter.Qs = np.linalg.cholesky(Q)
    filter.Rs = np.linalg.cholesky(R)
    
    # Predict
    filter.predict(dt, linear_fx)
    assert filter.x[0] == pytest.approx(0.1, abs=1e-5)
    
    # Update
    z = np.array([0.15])
    filter.update(z, linear_hx)
    assert len(filter.x) == 2
    assert np.all(np.isfinite(filter.P))

def test_enkf(linear_setup):
    dt, dim_x, dim_z, x0, P0, Q, R = linear_setup
    filter = EnKF(dim_x, dim_z, ensemble_size=20)
    filter.initialize_ensemble(x0, P0)
    filter.Q = Q
    filter.R = R
    
    filter.predict(dt, linear_fx)
    z = np.array([0.15])
    filter.update(z, linear_hx)
    assert len(filter.x) == 2
    assert filter.X.shape == (2, 20)

def test_ckf(linear_setup):
    dt, dim_x, dim_z, x0, P0, Q, R = linear_setup
    filter = CKF(dim_x, dim_z)
    filter.x = x0.copy()
    filter.P = P0.copy()
    filter.Q = Q
    filter.R = R
    
    filter.predict(dt, linear_fx)
    z = np.array([0.15])
    filter.update(z, linear_hx)
    assert len(filter.x) == 2

def test_pf(linear_setup):
    dt, dim_x, dim_z, x0, P0, Q, R = linear_setup
    filter = ParticleFilter(dim_x, dim_z, num_particles=100)
    filter.initialize_particles(x0, P0)
    filter.Q = Q
    filter.R = R
    
    filter.predict(dt, linear_fx)
    z = np.array([0.15])
    filter.update(z, linear_hx)
    assert len(filter.x) == 2
    assert filter.particles.shape == (100, 2)

def test_akf(linear_setup):
    dt, dim_x, dim_z, x0, P0, Q, R = linear_setup
    filter = AKF(dim_x, dim_z, window_size=5)
    filter.x = x0.copy()
    filter.P = P0.copy()
    filter.F = np.array([[1, dt], [0, 1]])
    filter.H = np.array([[1, 0]])
    
    # Run a few steps to fill window
    for _ in range(6):
        filter.predict()
        filter.update(np.array([0.1]))
        
    assert len(filter.x) == 2
    assert filter.R.shape == (1, 1)

def test_imm(linear_setup):
    dt, dim_x, dim_z, x0, P0, Q, R = linear_setup
    
    # Two models: one with identity F, one with CV F
    f1 = KF(dim_x, dim_z)
    f1.x = x0.copy()
    f1.P = P0.copy()
    f1.F = np.eye(2)
    f1.H = np.array([[1, 0]])
    
    f2 = KF(dim_x, dim_z)
    f2.x = x0.copy()
    f2.P = P0.copy()
    f2.F = np.array([[1, dt], [0, 1]])
    f2.H = np.array([[1, 0]])
    
    trans = np.array([[0.95, 0.05], [0.05, 0.95]])
    imm = IMM([f1, f2], trans)
    
    imm.predict(dt)
    imm.update(np.array([0.1]))
    
    assert len(imm.x) == 2
    assert len(imm.mu) == 2
    assert np.isclose(np.sum(imm.mu), 1.0)
