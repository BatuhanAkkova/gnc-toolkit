import pytest
import numpy as np
from gnc_toolkit.classical_control.pid import PID
from gnc_toolkit.classical_control.bdot import BDot
from gnc_toolkit.classical_control.momentum_dumping import CrossProductLaw

class TestPID:
    def test_proportional(self):
        pid = PID(kp=2.0, ki=0.0, kd=0.0)
        output = pid.update(error=1.0, dt=1.0)
        assert output == 2.0

    def test_integral(self):
        pid = PID(kp=0.0, ki=1.0, kd=0.0)
        pid.update(error=1.0, dt=1.0) # Integral = 1.0
        output = pid.update(error=1.0, dt=1.0) # Integral = 2.0
        assert output == 2.0

    def test_derivative(self):
        pid = PID(kp=0.0, ki=0.0, kd=1.0)
        pid.update(error=1.0, dt=1.0) # Prev error = 1.0
        # Next error 2.0, dt=1.0 -> derivative = (2-1)/1 = 1.0
        output = pid.update(error=2.0, dt=1.0)
        assert output == 1.0

    def test_saturation_and_anti_windup(self):
        # Limit output to [-10, 10]
        pid = PID(kp=1.0, ki=10.0, kd=0.0, output_limits=(-10.0, 10.0), anti_windup_method="clamping")
        
        # Huge error to cause saturation
        # P=100, I=100*1*10 = 1000. Total = 1100. Clamped to 10.
        output = pid.update(error=100.0, dt=1.0)
        assert output == 10.0
        
        # Check if integrator stopped (clamped)
        assert pid.integral_error == 0.0

    def test_reset(self):
        pid = PID(kp=1.0, ki=1.0, kd=1.0)
        pid.update(error=1.0, dt=1.0)
        pid.reset()
        assert pid.integral_error == 0.0
        assert pid.previous_error == 0.0

class TestBDot:
    def test_control_law(self):
        gain = 1000.0
        bdot = BDot(gain=gain)
        
        b_rate = np.array([1.0, 0.0, -1.0])
        expected_moment = -gain * b_rate # [-1000, 0, 1000]
        
        moment = bdot.calculate_control(b_rate)
        
        np.testing.assert_array_equal(moment, expected_moment)

    def test_discrete_calculation(self):
        gain = 1.0
        bdot = BDot(gain=gain)
        
        dt = 0.5
        b_prev = np.array([0.0, 0.0, 0.0])
        b_curr = np.array([1.0, 2.0, 3.0])
        
        # b_dot_est = ([1,2,3] - [0,0,0]) / 0.5 = [2, 4, 6]
        # m = -1 * [2, 4, 6] = [-2, -4, -6]
        
        moment = bdot.calculate_control_discrete(b_curr, b_prev, dt)
        expected = np.array([-2.0, -4.0, -6.0])
        
        np.testing.assert_array_almost_equal(moment, expected)

    def test_zero_dt(self):
        bdot = BDot(gain=1.0)
        moment = bdot.calculate_control_discrete(np.zeros(3), np.zeros(3), 0.0)
        np.testing.assert_array_equal(moment, np.zeros(3))

class TestCrossProductLaw:
    def test_control_law(self):
        gain = 0.5
        ctrl = CrossProductLaw(gain=gain)
        
        # h = [1, 0, 0], B = [0, 0, 1e-4]
        h = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 1e-4])
        
        moment = ctrl.calculate_control(h, b)
        expected = np.array([0.0, -5000.0, 0.0])
        
        np.testing.assert_array_almost_equal(moment, expected)

    def test_zero_field(self):
        ctrl = CrossProductLaw(gain=1.0)
        moment = ctrl.calculate_control([1,1,1], [0,0,0])
        np.testing.assert_array_equal(moment, np.zeros(3))
