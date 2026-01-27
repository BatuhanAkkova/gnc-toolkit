import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.integrators import RK4, RK45, RK853

@pytest.mark.parametrize("IntegratorClass", [RK4, RK45, RK853])
def test_linear_ode(IntegratorClass):
    """
    dy/dt = y
    y(0) = 1
    Analytical: y(t) = e^t
    """
    if IntegratorClass == RK45:
        integrator = IntegratorClass(rtol=1e-8, atol=1e-10)
    elif IntegratorClass == RK853:
        integrator = IntegratorClass(rtol=1e-10, atol=1e-12)
    else:
        integrator = IntegratorClass()
    
    def f(t, y):
        return y
    
    t_span = (0, 1.0)
    y0 = [1.0]
    
    t_val, y_val = integrator.integrate(f, t_span, y0, dt=0.01)
    
    y_final = y_val[-1][0]
    y_expected = np.exp(1.0)
    
    # RK4 should be decently accurate
    if IntegratorClass == RK4:
        tol = 1e-5
    else:
        tol = 1e-6
        
    assert np.abs(y_final - y_expected) < tol

@pytest.mark.parametrize("IntegratorClass", [RK4, RK45, RK853])
def test_harmonic_oscillator(IntegratorClass):
    """
    y'' = -y
    State: x = [y, y']
    dx/dt = [y', -y]
    x(0) = [0, 1]
    Analytical: y(t) = sin(t), y'(t) = cos(t)
    """
    integrator = IntegratorClass()
    
    def f(t, x):
        return np.array([x[1], -x[0]])
    
    t_span = (0, 2 * np.pi)
    y0 = [0.0, 1.0]
    
    t_val, y_val = integrator.integrate(f, t_span, y0, dt=0.1)
    
    y_final = y_val[-1]
    y_expected = [0.0, 1.0]
    
    if IntegratorClass == RK4:
        tol = 1e-3
    else:
        tol = 1e-4
        
    np.testing.assert_allclose(y_final, y_expected, atol=tol)

def test_rk45_adaptive():
    """
    Test that RK45 adapts step size.
    dy/dt = -y (decay)
    """
    integrator = RK45(rtol=1e-6, atol=1e-8)
    
    def f(t, y):
        return -y
    
    t_span = (0, 10.0)
    y0 = [1.0]
    
    t_val, _ = integrator.integrate(f, t_span, y0, dt=1.0) # Start with large dt
        
    def f_steep(t, y):
        return 10 * y
        
    t_span = (0, 1.0)
    y0 = [1e-5]
    
    t_vals, _ = integrator.integrate(f_steep, t_span, y0, dt=0.1)
    
    # Steps should vary
    dts = np.diff(t_vals)
    assert not np.allclose(dts, dts[0])

def test_rk853_precision():
    """
    Test RK853 high precision.
    """
    # Using tighter tolerances
    integrator = RK853(rtol=1e-12, atol=1e-12)
    
    def f(t, y):
        return y
    
    t_span = (0, 1.0)
    y0 = [1.0]
    
    _, y_val = integrator.integrate(f, t_span, y0, dt=0.1)
    
    y_final = y_val[-1][0]
    y_expected = np.exp(1.0)
    
    assert np.abs(y_final - y_expected) < 1e-11
