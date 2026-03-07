import numpy as np
import pytest
from gnc_toolkit.kalman_filters.kf import KF
from gnc_toolkit.kalman_filters.fixed_interval_smoother import fixed_interval_smoother

def test_fixed_interval_smoother():
    """Test Fixed-Interval Smoother with a linear constant velocity model."""
    dt = 0.5
    num_steps = 30
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.01
    R = np.eye(1) * 0.1
    
    true_x = np.zeros((num_steps, 2))
    z = np.zeros((num_steps, 1))
    
    # Generate truth and measurements
    x = np.array([0.0, 1.0])
    for k in range(num_steps):
        true_x[k] = x
        z[k] = np.dot(H, x) + np.random.normal(0, np.sqrt(R[0,0]))
        x = np.dot(F, x) + np.random.multivariate_normal(np.zeros(2), Q)
        
    # Forward Pass (KF)
    kf = KF(dim_x=2, dim_z=1)
    kf.F = F
    kf.H = H
    kf.Q = Q
    kf.R = R
    
    xs_filt = []
    ps_filt = []
    
    for k in range(num_steps):
        kf.predict()
        kf.update(z[k])
        xs_filt.append(kf.x.copy())
        ps_filt.append(kf.P.copy())
        
    # Smoother Arguments
    Fs = [F] * (num_steps - 1)
    Qs = [Q] * (num_steps - 1)
    Zs = [zi for zi in z]
    Hs = [H] * num_steps
    Rs = [R] * num_steps
    
    # Fixed-Interval Smoothing
    xs_smooth, ps_smooth = fixed_interval_smoother(xs_filt, ps_filt, Fs, Qs, Zs, Hs, Rs)
    
    # Calculate RMSE
    rmse_filt = np.sqrt(np.mean((np.array(xs_filt)[:, 0] - true_x[:, 0])**2))
    rmse_smooth = np.sqrt(np.mean((xs_smooth[:, 0] - true_x[:, 0])**2))
    
    # Smoothed error should be lower than filtered error (statistically)
    # Using a slightly higher margin for stochasticity
    assert rmse_smooth <= rmse_filt * 1.05
    assert xs_smooth.shape == (num_steps, 2)
    
    # Verify covariance reduction
    for k in range(num_steps):
        assert np.trace(ps_smooth[k]) <= np.trace(ps_filt[k]) * 1.01
