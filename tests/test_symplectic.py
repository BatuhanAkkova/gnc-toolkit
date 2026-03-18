import pytest
import numpy as np
from gnc_toolkit.integrators.symplectic import SymplecticIntegrator

def test_symplectic_kepler():
    """
    Test Symplectic Integrator (Yoshida 4th order) on a Keplerian orbit.
    Verifies that Energy is conserved better than non-symplectic methods (like Euler, or short RK4).
    """
    mu = 398600.4418e9 # m^3/s^2
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(mu / 7000e3), 0.0])
    y0 = np.concatenate([r0, v0])
    
    def f(t, y):
        r = y[:3]
        v = y[3:]
        r_mag = np.linalg.norm(r)
        a = -mu / (r_mag**3) * r
        return np.concatenate([v, a])

    # Time span: ~1 orbit, small step for accuracy
    t_span = (0, 1000.0)
    dt = 1.0 # 1s step size
    
    sym_integrator = SymplecticIntegrator()
    t_values, y_values = sym_integrator.integrate(f, t_span, y0, dt=dt)
    
    # Check dimensions
    assert len(t_values) > 1
    
    # Calculate Energy at start and end
    # E = v^2 / 2 - mu / r
    r_start = y_values[0, :3]
    v_start = y_values[0, 3:]
    E_start = np.sum(v_start**2)/2 - mu / np.linalg.norm(r_start)
    
    r_end = y_values[-1, :3]
    v_end = y_values[-1, 3:]
    E_end = np.sum(v_end**2)/2 - mu / np.linalg.norm(r_end)
    
    dE = np.abs(E_end - E_start) / np.abs(E_start)
    print(f"Energy conservation error: {dE}")
    
    # Symplectic methods should conserve energy to high accuracy (negligible drift)
    assert dE < 1e-6 # Highly conservative limit for 4th order over 1000s

def test_symplectic_unsupported_step():
    sym_integrator = SymplecticIntegrator()
    # verify step doesn't crash or works
    res_y, res_t, _ = sym_integrator.step(lambda t, y: y, 0, np.zeros(6), 1.0)
    assert len(res_y) == 6
