import numpy as np
import pytest
from gnc_toolkit.kalman_filters.ekf import EKF
from gnc_toolkit.kalman_filters.kf import KF
from gnc_toolkit.kalman_filters.mekf import MEKF
from gnc_toolkit.kalman_filters.ukf import UKF, UKF_Attitude
from gnc_toolkit.utils.quat_utils import quat_mult, quat_normalize, axis_angle_to_quat, quat_rot, quat_conj
from gnc_toolkit.sensors.sun_sensor import SunSensor

def test_kf_initialization():
    dim_x = 2
    dim_z = 1
    kf = KF(dim_x, dim_z)
    assert kf.x.shape == (dim_x,)
    assert kf.P.shape == (dim_x, dim_x)
    assert kf.F.shape == (dim_x, dim_x)
    assert kf.H.shape == (dim_z, dim_x)
    assert kf.R.shape == (dim_z, dim_z)
    assert kf.Q.shape == (dim_x, dim_x)

def test_kf_constant_velocity():
    """Test KF with a simple 1D constant velocity model."""
    dt = 0.1
    kf = KF(dim_x=2, dim_z=1)
    
    # State: [position, velocity]
    kf.F = np.array([[1, dt], [0, 1]])
    kf.H = np.array([[1, 0]])
    kf.P *= 10
    kf.R *= 0.1
    kf.Q = np.array([[0.001, 0], [0, 0.001]])

    true_x = np.array([0., 1.0]) # Pos=0, Vel=1 m/s
    
    # Run filter for 20 steps
    for _ in range(20):
        # Simulate Truth
        true_x = np.dot(kf.F, true_x)
        
        # Simulate Measurement
        z = np.dot(kf.H, true_x) + np.random.normal(0, np.sqrt(kf.R[0,0]))
        
        # Filter Cycle
        kf.predict()
        kf.update(z)
        
    error_pos = np.abs(true_x[0] - kf.x[0])
    error_vel = np.abs(true_x[1] - kf.x[1])
    
    # Relaxed thresholds for stochastic test
    assert error_pos < 1.0
    assert error_vel < 0.5

def test_ekf_nonlinear_tracking():
    """Test EKF with a non-linear measurement model (measuring squared state)."""
    dt = 0.1
    ekf = EKF(dim_x=1, dim_z=1)
    
    # Process Model: x_k+1 = x_k + vel * dt
    def f_func(x, dt, u, **kwargs): return x + 1.0 * dt
    def f_jac(x, dt, u, **kwargs): return np.array([[1.0]])
    
    # Measurement Model: z = x^2
    def h_func(x, **kwargs): return x**2
    def h_jac(x, **kwargs): return np.array([[2*x[0]]])
    
    ekf.P *= 1.0
    ekf.R *= 0.01
    ekf.Q *= 0.001
    ekf.x = np.array([1.0])
    
    true_x = 1.0
    
    for _ in range(20):
        true_x += 1.0 * dt
        z = true_x**2 + np.random.normal(0, 0.1)
        
        ekf.predict(f_func, f_jac, dt)
        ekf.update(np.array([z]), h_func, h_jac)
        
    error = np.abs(true_x - ekf.x[0])
    assert error < 0.5

def test_mekf_attitude_tracking():
    """Test MEKF for attitude estimation."""
    dt = 0.1
    omega_body = np.array([0.1, 0.05, 0.2]) # Constant body rate
    z_ref_inertial = np.array([1.0, 0.0, 0.0])
    
    q0 = np.array([0, 0, 0, 1.0])
    mekf = MEKF(q_init=q0)
    
    mekf.Q = np.eye(6) * 0.0001
    mekf.R = np.eye(3) * 0.01
    
    q_true = q0.copy()
    
    # Use real sun sensor
    sun_sensor = SunSensor(noise_std=0.01)
    
    for _ in range(50):
        # Truth Update
        dq_true = axis_angle_to_quat(omega_body * dt)
        q_true = quat_normalize(quat_mult(q_true, dq_true))
        
        # Measurement simulation using SunSensor
        z_body_true = quat_rot(quat_conj(q_true), z_ref_inertial)
        z_body_meas = sun_sensor.measure(z_body_true)
        
        # Filter Step
        mekf.predict(omega_body, dt)
        mekf.update(z_body_meas, z_ref_inertial)
        
    error = 1.0 - np.abs(np.dot(q_true, mekf.q))
    assert error < 1e-3

def test_ukf_nonlinear_tracking():
    """Test UKF with non-linear model (vector state version)."""
    dt = 0.1
    ukf = UKF(dim_x=1, dim_z=1)
    
    # Process: x_k+1 = x_k + vel * dt
    def f_func(x, dt): return x + 1.0 * dt
    
    # Measurement: z = x^2
    def h_func(x): return x**2
    
    ukf.P *= 1.0
    ukf.R *= 0.01
    ukf.Q *= 0.001
    ukf.x = np.array([1.0])
    
    true_x = 1.0
    
    for _ in range(20):
        true_x += 1.0 * dt
        z = true_x**2 + np.random.normal(0, 0.1)
        
        ukf.predict(dt, f_func)
        ukf.update(np.array([z]), h_func)
        
    error = np.abs(true_x - ukf.x[0])
    assert error < 0.5

def test_ukf_attitude_initialization():
    ukf = UKF_Attitude()
    assert ukf.x.shape == (7,) 
    assert ukf.P.shape == (6, 6) 
    
    assert np.isclose(np.linalg.norm(ukf.x[:4]), 1.0)

def test_ukf_attitude_prediction():
    ukf = UKF_Attitude()
    dt = 0.1
    
    ukf.x = np.array([0., 0., 0., 1., 0.1, 0.1, 0.1]) 
    
    def fx(x, dt, omega_meas):
        q = x[:4]
        bias = x[4:]
        
        omega = omega_meas - bias
        
        omega_norm = np.linalg.norm(omega)
        if omega_norm > 1e-10:
            axis = omega / omega_norm
            angle = omega_norm * dt
            dq = axis_angle_to_quat(axis * angle)
            q_new = quat_mult(q, dq)
            q_new = quat_normalize(q_new)
        else:
            q_new = q
            
        return np.concatenate([q_new, bias])
    
    omega_meas = np.array([0.1, 0.1, 0.1]) 
    
    ukf.predict(dt, fx, omega_meas=omega_meas)
    
    assert np.allclose(ukf.x[:4], np.array([0, 0, 0, 1]), atol=1e-5)
    
    assert np.allclose(ukf.x[4:], np.array([0.1, 0.1, 0.1]), atol=1e-5)

def test_ukf_attitude_update():
    ukf = UKF_Attitude()
    ukf.P *= 0.1
    ukf.R *= 0.001
    
    angle = np.pi/2
    q_true = np.array([0, 0, np.sin(angle/2), np.cos(angle/2)])
    
    z_ref = np.array([1.0, 0.0, 0.0])
    
    # Use real sun sensor
    sensor = SunSensor(noise_std=0.0)
    
    q_conj = quat_conj(q_true)
    z_body_true = quat_rot(q_conj, z_ref)
    
    z_meas = sensor.measure(z_body_true)
    
    def hx(x, z_ref):
        q = x[:4]
        q_conj = quat_conj(q)
        z_pred = quat_rot(q_conj, z_ref)
        return z_pred
        
    ukf.update(z_meas, hx, z_ref=z_ref)
    
    assert ukf.x.shape == (7,)
    assert np.isclose(np.linalg.norm(ukf.x[:4]), 1.0)
    pass
