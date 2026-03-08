import numpy as np
import pytest
from gnc_toolkit.classical_control import CrossProductLaw, RateDampingControl

def test_cross_product_law():
    # Gain: 100, Max dipole: 0.1 Am^2
    controller = CrossProductLaw(gain=0.1, max_dipole=0.1) # gain is usually small for desat
    
    # h error along X, B along Y
    h_error = np.array([0.1, 0.0, 0.0]) # 0.1 Nms
    b_field = np.array([0.0, 50e-6, 0.0]) # 50 microTesla
    
    # m = k * (h x b) / |b|^2
    # h x b = [0, 0, 0.1 * 50e-6] = [0, 0, 5e-6]
    # |b|^2 = (50e-6)^2 = 2500e-12 = 2.5e-9
    # m = (0.1 / 2.5e-9) * [0, 0, 5e-6] = 4e7 * [0, 0, 5e-6] = [0, 0, 200]
    
    m = controller.calculate_control(h_error, b_field)
    
    # Should be saturated to 0.1
    assert np.linalg.norm(m) == pytest.approx(0.1)
    assert m[2] == pytest.approx(0.1)

def test_rate_damping():
    controller = RateDampingControl(gain=2.0, max_torque=0.5)
    
    omega = np.array([0.1, 0.2, 0.0])
    
    # T = -2 * [0.1, 0.2, 0] = [-0.2, -0.4, 0]
    torque = controller.compute_torque(omega)
    
    assert torque[0] == pytest.approx(-0.2)
    assert torque[1] == pytest.approx(-0.4)
    assert torque[2] == 0.0
    
    # Test saturation
    omega_large = np.array([10.0, 0.0, 0.0])
    # T = -20.0, should be saturated to 0.5
    torque_sat = controller.compute_torque(omega_large)
    
    assert np.linalg.norm(torque_sat) == pytest.approx(0.5)
    assert torque_sat[0] == -0.5
