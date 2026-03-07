import numpy as np
import pytest
from gnc_toolkit.attitude_dynamics.flexible_body import flexible_body_dynamics, coupled_flexible_rigid_dynamics

def test_flexible_body_oscillation():
    # Single mode: mass-spring-damper
    # eta_ddot + 2*zeta*wn*eta_dot + wn^2*eta = 0 (omega_dot = 0)
    eta = np.array([1.0])
    eta_dot = np.array([0.0])
    omega_dot = np.array([0.0, 0.0, 0.0])
    natural_freqs = np.array([10.0]) # 10 rad/s
    damping_ratios = np.array([0.05]) # 5% damping
    modal_influence = np.array([[0.0, 0.0, 0.0]])
    
    eta_ddot = flexible_body_dynamics(eta, eta_dot, omega_dot, natural_freqs, damping_ratios, modal_influence)
    
    # eta_ddot = -wn^2 * eta = -100 * 1 = -100
    assert np.isclose(eta_ddot[0], -100.0)

def test_coupled_dynamics_conservation():
    # Zero torque, zero damping
    J_rigid = np.eye(3)
    omega = np.array([0.1, 0.0, 0.0])
    torque = np.zeros(3)
    eta = np.array([1.0])
    eta_dot = np.zeros(1)
    natural_freqs = np.array([10.0])
    damping_ratios = np.array([0.0])
    modal_influence = np.array([[0.1, 0.0, 0.0]]) # Coupling on X axis
    
    omega_dot, eta_ddot = coupled_flexible_rigid_dynamics(J_rigid, omega, torque, eta, eta_dot, natural_freqs, damping_ratios, modal_influence)
    
    # The system should respond to the coupling
    assert not np.allclose(omega_dot, 0.0)
    assert not np.allclose(eta_ddot, 0.0)

if __name__ == "__main__":
    test_flexible_body_oscillation()
    test_coupled_dynamics_conservation()
    print("Test passed!")
