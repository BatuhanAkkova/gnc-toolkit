import numpy as np
import pytest

from gnc_toolkit.optimal_control.h2_control import H2Controller
from gnc_toolkit.optimal_control.mpc_casadi import CasadiNMPC

def test_h2_controller():
    # Simple 1D system
    A = np.array([[1.0]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    Q_lqr = np.array([[1.0]])
    R_lqr = np.array([[1.0]])
    Q_lqe = np.array([[1.0]])
    R_lqe = np.array([[1.0]])
    
    controller = H2Controller(A, B, C, Q_lqr, R_lqr, Q_lqe, R_lqe)
    K, L = controller.solve()
    
    # Both gains should be > 0
    assert K[0, 0] > 0
    assert L[0, 0] > 0


def test_casadi_nmpc():
    # Attempt to skip if casadi is not fully available, but should be via toolkit deps
    nx = 1
    nu = 1
    N = 3
    dt = 0.1
    
    # Simple integrator dx/dt = u
    def dynamics(x, u):
        return u
        
    def stage_cost(x, u):
        import casadi as ca
        return x**2 + u**2
        
    def term_cost(x):
        return x**2
        
    nmpc = CasadiNMPC(
        nx=nx, nu=nu, horizon=N, dt=dt,
        dynamics_func=dynamics,
        cost_func=stage_cost,
        terminal_cost_func=term_cost,
        u_min=-1.0, u_max=1.0,
        x_min=-10.0, x_max=10.0,
        discrete=False
    )
    
    u_opt = nmpc.solve(x0=np.array([1.0]))
    assert u_opt.shape == (N, nu)
    # The optimal input to regulate from 1.0 to 0 should be negative
    assert u_opt[0, 0] < 0
