import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import pytest
import time

from opengnc.fdir.residual_generation import ObserverResidualGenerator, AnalyticalRedundancy
from opengnc.fdir.parity_space import ParitySpaceDetector
from opengnc.fdir.safe_mode import SafeModeLogic, SafeModeCondition, SystemMode
from opengnc.fdir.failure_accommodation import ActuatorAccommodation

def test_observer_residual_generator():
    A = np.array([[1.0]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = None
    L = np.array([[0.5]])
    
    obs = ObserverResidualGenerator(A, B, C, D, L)
    
    u = np.array([1.0])
    y = np.array([0.0])
    
    r = obs.step(u, y)
    assert np.allclose(r, [[0.0]])
    
    y_next = np.array([1.0])
    r_next = obs.step(u, y_next)
    assert np.allclose(r_next, [[0.0]])
    
    y_fault = np.array([3.0])
    r_fault = obs.step(u, y_fault)
    assert np.allclose(r_fault, [[1.0]])

def test_parity_space():
    M = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 2.0, 3.0],
        [2.0, -1.0, 1.0]
    ])
    
    detector = ParitySpaceDetector(M)
    
    x = np.array([1.0, 2.0, 3.0])
    y = M @ x
    
    p_vec = detector.get_parity_vector(y)
    assert np.allclose(p_vec, 0.0, atol=1e-10)
    assert not detector.detect_fault(y, threshold=0.1)
    
    y_fault = y.copy()
    y_fault[0] += 5.0
    
    assert detector.detect_fault(y_fault, threshold=0.1)
    isolated_idx = detector.isolate_fault(y_fault)
    assert isolated_idx == 0

    y_fault2 = y.copy()
    y_fault2[3] += -2.0
    assert detector.detect_fault(y_fault2, threshold=0.1)
    assert detector.isolate_fault(y_fault2) == 3


def test_safe_mode():
    logic = SafeModeLogic()
    
    x_is_bad = False
    cond = SafeModeCondition(lambda: x_is_bad, trigger_time_sec=0.1)
    logic.add_condition("bad_x", cond)
    
    from unittest.mock import patch
    with patch('time.time') as mock_time:
        mock_time.return_value = 1000.0
        assert logic.update() == SystemMode.NOMINAL
        
        x_is_bad = True
        assert logic.update() == SystemMode.NOMINAL
        
        mock_time.return_value = 1000.15
        assert logic.update() == SystemMode.SAFE
        
        assert len(logic.history) == 1
        assert "bad_x" in logic.history[0]["reason"]
        logic_safe = SafeModeLogic(initial_mode=SystemMode.SAFE)
        assert logic_safe.update() == SystemMode.SAFE
        
        logic_safe.force_mode(SystemMode.RECOVERY, reason="Manual")
        assert logic_safe.mode == SystemMode.RECOVERY
        assert len(logic_safe.history) == 1
        assert logic_safe.history[0]["reason"] == "Manual"

def test_actuator_accommodation():
    B = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0]
    ])
    
    acc = ActuatorAccommodation(B)
    tau = np.array([1.0, 1.0])
    
    u_nominal = acc.allocate(tau)
    assert np.allclose(B @ u_nominal, tau.reshape(-1, 1))
    
    acc.set_health(2, 0.0)
    u_fault = acc.allocate(tau)
    
    assert np.allclose(u_fault[2], 0.0, atol=1e-5)
    assert np.allclose(B @ u_fault, tau.reshape(-1, 1))

def test_actuator_accommodation_extended():
    B = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0]
    ])
    
    W_init_2d = np.diag([2.0, 3.0, 4.0])
    acc_2d = ActuatorAccommodation(B, W_init_2d)
    assert np.allclose(acc_2d.W_diag, [2.0, 3.0, 4.0])
    
    acc_1d = ActuatorAccommodation(B, np.array([2.0, 3.0, 4.0]))
    assert np.allclose(acc_1d.W_diag, [2.0, 3.0, 4.0])
    
    with pytest.raises(IndexError):
        acc_1d.set_health(5, 1.0)
        
    B_singular = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    acc_singular = ActuatorAccommodation(B_singular)
    B_pinv = acc_singular.update_allocation_matrix()
    assert np.allclose(B_pinv, np.linalg.pinv(B_singular))

def test_parity_space_extended():
    M_non_redundant = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        ParitySpaceDetector(M_non_redundant)
        
    M = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])
    detector = ParitySpaceDetector(M)
    y_no_fault = M @ np.array([1.0, 2.0])
    assert detector.isolate_fault(y_no_fault) == -1
    
    M_zero_col = np.array([
        [1.0],
        [0.0],
        [0.0]
    ])
    detector_zero = ParitySpaceDetector(M_zero_col)
    y_fault = np.array([1.0, 1.0, 0.0]) # fault in sensor 1
    isolated = detector_zero.isolate_fault(y_fault)
    assert isolated >= 0

def test_analytical_redundancy():
    assert AnalyticalRedundancy.check_threshold(np.array([2.0]), 1.0)
    assert not AnalyticalRedundancy.check_threshold(np.array([0.5]), 1.0)
    
    q1 = np.array([1, 0, 0, 0])
    q2 = np.array([0, 1, 0, 0])
    res = AnalyticalRedundancy.gyro_vs_quaternion_residual(q1, q2)
    assert np.allclose(res, q1 - q2)




