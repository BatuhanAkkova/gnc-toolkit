import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import pytest
import time

from gnc_toolkit.fdir.residual_generation import ObserverResidualGenerator, AnalyticalRedundancy
from gnc_toolkit.fdir.parity_space import ParitySpaceDetector
from gnc_toolkit.fdir.safe_mode import SafeModeLogic, SafeModeCondition, SystemMode
from gnc_toolkit.fdir.failure_accommodation import ActuatorAccommodation

def test_observer_residual_generator():
    # Simple 1D system: x_{k+1} = x_k + u_k, y_k = x_k
    A = np.array([[1.0]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = None
    L = np.array([[0.5]])  # observer gain
    
    obs = ObserverResidualGenerator(A, B, C, D, L)
    
    # Step 1: No fault
    u = np.array([1.0])
    y = np.array([0.0])  # initial state is 0, so x_hat=0, y_hat=0
    
    r = obs.step(u, y)
    assert np.allclose(r, [[0.0]])  # y - y_hat = 0 - 0 = 0
    
    # Next state x_hat should be A*x_hat + B*u + L*r = 0 + 1 + 0 = 1
    # Step 2: Measurements match prediction
    y_next = np.array([1.0])
    r_next = obs.step(u, y_next)
    assert np.allclose(r_next, [[0.0]])  # y - y_hat = 1 - 1 = 0
    
    # Step 3: Inject fault (sensor reads 2.0 instead of 2.0)
    # x_hat is updated to 2 (from previous step 1.0 + 1.0)
    y_fault = np.array([3.0])  # reading is higher
    r_fault = obs.step(u, y_fault)
    assert np.allclose(r_fault, [[1.0]])  # 3.0 - 2.0 = 1.0

def test_parity_space():
    # 5 sensors for 3D state (2 redundant)
    # This allows for Isolation (requires p-n >= 2)
    M = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 2.0, 3.0],
        [2.0, -1.0, 1.0]
    ])
    
    detector = ParitySpaceDetector(M)
    
    # Test case 1: No fault
    x = np.array([1.0, 2.0, 3.0])
    y = M @ x
    
    p_vec = detector.get_parity_vector(y)
    assert np.allclose(p_vec, 0.0, atol=1e-10)
    assert not detector.detect_fault(y, threshold=0.1)
    
    # Test case 2: Fault in sensor 0 (bias)
    y_fault = y.copy()
    y_fault[0] += 5.0
    
    assert detector.detect_fault(y_fault, threshold=0.1)
    isolated_idx = detector.isolate_fault(y_fault)
    assert isolated_idx == 0

    # Test case 3: Fault in sensor 3
    y_fault2 = y.copy()
    y_fault2[3] += -2.0
    assert detector.detect_fault(y_fault2, threshold=0.1)
    assert detector.isolate_fault(y_fault2) == 3


def test_safe_mode():
    logic = SafeModeLogic()
    
    # Condition: fail if x is bad
    x_is_bad = False
    cond = SafeModeCondition(lambda: x_is_bad, trigger_time_sec=0.1)
    logic.add_condition("bad_x", cond)
    
    assert logic.update() == SystemMode.NOMINAL
    
    # Trigger condition
    x_is_bad = True
    assert logic.update() == SystemMode.NOMINAL  # Duration not met yet
    
    time.sleep(0.15)
    assert logic.update() == SystemMode.SAFE
    
    # Check history
    assert len(logic.history) == 1
    assert "bad_x" in logic.history[0]["reason"]

def test_actuator_accommodation():
    # 2 DOF system, 3 actuators
    B = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0]
    ])
    
    acc = ActuatorAccommodation(B)
    tau = np.array([1.0, 1.0])
    
    # Nominal allocation
    u_nominal = acc.allocate(tau)
    # B * u should equal tau
    assert np.allclose(B @ u_nominal, tau.reshape(-1, 1))
    
    # Fail actuator 2 (index 2)
    acc.set_health(2, 0.0)
    u_fault = acc.allocate(tau)
    
    # Should be close to [1.0, 1.0, 0.0]^T roughly speaking since 2 is failed
    assert np.allclose(u_fault[2], 0.0, atol=1e-5)
    assert np.allclose(B @ u_fault, tau.reshape(-1, 1))

if __name__ == "__main__":
    pytest.main([__file__])
