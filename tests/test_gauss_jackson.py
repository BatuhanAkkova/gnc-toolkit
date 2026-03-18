import pytest
import numpy as np
from gnc_toolkit.integrators.ab_moulton import AdamsBashforthMoultonIntegrator

def test_ab_moulton_kepler():
    """
    Test Adams-Bashforth-Moulton 8th order integrator on a Keplerian orbit.
    Compare with Analytical.
    """
    # Constants
    mu = 398600.4418e9 # m^3/s^2
    
    # Initial state (circular orbit at 7000 km)
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(mu / 7000e3), 0.0])
    y0 = np.concatenate([r0, v0])
    
    # Equations of motion
    def f(t, y):
        r = y[:3]
        v = y[3:]
        r_mag = np.linalg.norm(r)
        a = -mu / (r_mag**3) * r
        return np.concatenate([v, a])

    # Time span: ~1 orbit (approx 5800s)
    t_span = (0, 1000.0)
    dt = 10.0 # 10s step size
    omega = np.sqrt(mu / 7000e3**3)

    ab_integrator = AdamsBashforthMoultonIntegrator()
    t_values, y_values = ab_integrator.integrate(f, t_span, y0, dt=dt)
    
    # Analytical Solution at end time
    t_end = t_span[1]
    r_exact = np.array([7000e3 * np.cos(omega * t_end), 7000e3 * np.sin(omega * t_end), 0.0])
    v_exact = np.array([-7000e3 * omega * np.sin(omega * t_end), 7000e3 * omega * np.cos(omega * t_end), 0.0])
    
    # Compare final state with Exact
    r_ab_f = y_values[-1, :3]
    v_ab_f = y_values[-1, 3:]
    
    error_r = np.linalg.norm(r_ab_f - r_exact)
    error_v = np.linalg.norm(v_ab_f - v_exact)
    
    print(f"AB Position Error: {error_r} m")
    print(f"AB Velocity Error: {error_v} m/s")
    
    # Verify accurate to meters (expecting < 1e-3)
    assert error_r < 1e-3
    assert error_v < 1e-3

def test_ab_moulton_unsupported_step():
    ab_integrator = AdamsBashforthMoultonIntegrator()
    with pytest.raises(NotImplementedError):
        ab_integrator.step(None, 0, None, 10.0)
