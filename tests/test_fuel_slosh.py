import numpy as np
import pytest
from gnc_toolkit.attitude_dynamics.fuel_slosh import fuel_slosh_dynamics, fuel_slosh_torque

def test_slosh_period():
    # Simple pendulum period: T = 2*pi * sqrt(L/g)
    # L = 1.0, g = 9.81 => T = 2.006 s
    # Natural frequency: wn = sqrt(g/L) = 3.132 rad/s
    # Small angle theta_ddot = - (g/L) * theta
    
    L = 1.0
    r_base = np.zeros(3)
    g_equiv = np.array([0, 0, -9.81])
    omega = np.zeros(3)
    omega_dot = np.zeros(3)
    theta = 0.1
    theta_dot = 0.0
    
    theta_ddot = fuel_slosh_dynamics(theta, theta_dot, omega, omega_dot, L, r_base, g_equiv)
    
    # Expected: -sin(0.1) * 9.81 / 1.0 approx -0.981
    assert np.isclose(theta_ddot, -np.sin(0.1) * 9.81, atol=1e-3)

def test_slosh_torque_reverses():
    m_p = 5.0
    L = 1.0
    theta = 0.1
    theta_dot = 0.0
    theta_ddot = -0.981
    r_base = np.array([0, 0, -1.0])
    
    torque = fuel_slosh_torque(m_p, L, theta, theta_dot, theta_ddot, r_base)
    
    # Slosh torque should be non-zero for displaced pendulum
    assert not np.allclose(torque, 0.0)
    assert torque.shape == (3,)
