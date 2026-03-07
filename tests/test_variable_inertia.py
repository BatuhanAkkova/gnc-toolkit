import numpy as np
import pytest
from gnc_toolkit.attitude_dynamics.variable_inertia import variable_inertia_euler_equations, mass_depletion_J_dot

def test_variable_inertia_acceleration():
    # Case where J_dot reverses the effects of torque or gyro
    J = np.eye(3)
    J_dot = np.eye(3) * 0.1 # Increasing inertia
    omega = np.array([1.0, 0.0, 0.0])
    torque = np.zeros(3)
    
    omega_dot = variable_inertia_euler_equations(J, J_dot, omega, torque)
    
    # J*omega_dot + J_dot*omega + omega x H = 0
    # I*omega_dot + 0.1*I*[1,0,0] + 0 = 0
    # omega_dot = [-0.1, 0, 0]
    assert np.isclose(omega_dot[0], -0.1)

def test_mass_depletion_J_dot():
    dm_dt = -0.1 # 0.1 kg/s loss
    r_point = np.array([1.0, 0, 0])
    
    J_dot = mass_depletion_J_dot(None, 10.0, dm_dt, r_point)
    
    # r_sq = 1.0, r_outer = [[1,0,0],[0,0,0],[0,0,0]]
    # (r_sq*I - r_outer) = [[0,0,0],[0,1,0],[0,0,1]]
    # J_dot = -0.1 * [[0,0,0],[0,1,0],[0,0,1]]
    assert np.isclose(J_dot[0, 0], 0.0)
    assert np.isclose(J_dot[1, 1], -0.1)
    assert np.isclose(J_dot[2, 2], -0.1)
