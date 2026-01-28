import pytest
import numpy as np
from gnc_toolkit.sensors.star_tracker import StarTracker
from gnc_toolkit.sensors.sun_sensor import SunSensor
from gnc_toolkit.sensors.magnetometer import Magnetometer
from gnc_toolkit.sensors.gyroscope import Gyroscope
from gnc_toolkit.actuators.reaction_wheel import ReactionWheel
from gnc_toolkit.actuators.magnetorquer import Magnetorquer
from gnc_toolkit.actuators.thruster import Thruster, ChemicalThruster, ElectricThruster

class TestSensors:
    def test_star_tracker(self):
        st = StarTracker(noise_std=0.0, bias=np.array([0.1, 0, 0]))
        true_quat = np.array([0.0, 0.0, 0.0, 1.0]) # [x, y, z, w]
        measured = st.measure(true_quat)
        
        # With bias 0.1 rad in x, the measured quat should reflect that.
        # q_err [sin(0.05), 0, 0, cos(0.05)] = [0.05, 0, 0, 0.998]
        assert not np.allclose(measured, true_quat)
        assert measured[0] > 0 # x component should be positive
        
        # Test noise (rough check)
        st_noisy = StarTracker(noise_std=0.01)
        m1 = st_noisy.measure(true_quat)
        m2 = st_noisy.measure(true_quat)
        assert not np.allclose(m1, m2)

    def test_sun_sensor(self):
        ss = SunSensor(noise_std=0.0)
        vec = np.array([1.0, 0.0, 0.0])
        meas = ss.measure(vec)
        assert np.allclose(meas, vec)
        
        ss_noisy = SunSensor(noise_std=0.1)
        meas_noisy = ss_noisy.measure(vec)
        # Should be normalized
        assert np.isclose(np.linalg.norm(meas_noisy), 1.0)

    def test_magnetometer(self):
        mag = Magnetometer(bias=np.array([1e-6, 0, 0]))
        true_b = np.array([20e-6, 0, 0])
        meas = mag.measure(true_b)
        assert np.allclose(meas, np.array([21e-6, 0, 0]))

    def test_gyroscope(self):
        # Test Bias Instability
        gyro = Gyroscope(noise_std=0.0, bias_stability=0.01, dt=1.0)
        w_true = np.zeros(3)
        
        b0 = gyro.current_bias.copy()
        m1 = gyro.measure(w_true)
        b1 = gyro.current_bias.copy()
        
        # Bias should have changed (random walk)
        assert not np.allclose(b0, b1)
        # Measurement should equal new bias (since true w is 0 and noise is 0)
        assert np.allclose(m1, b1)

class TestActuators:
    def test_reaction_wheel(self):
        rw = ReactionWheel(max_torque=0.1, max_momentum=1.0, inertia=0.1)
        
        # Normal op
        assert rw.command(0.05) == 0.05
        
        # Saturation
        assert rw.command(0.2) == 0.1
        assert rw.command(-0.2) == -0.1
        
        # Momentum saturation
        # If speed is 10 rad/s -> momentum = 1.0 (max)
        # Positive torque should be blocked
        assert rw.command(0.05, current_speed=10.0) == 0.0
        # Negative torque allowed (braking)
        assert rw.command(-0.05, current_speed=10.0) == -0.05

    def test_magnetorquer(self):
        mtq = Magnetorquer(max_dipole=10.0)
        assert mtq.command(5.0) == 5.0
        assert mtq.command(15.0) == 10.0
        assert mtq.command(-12.0) == -10.0

    def test_thruster(self):
        thr = Thruster(max_thrust=5.0)
        assert thr.command(2.0) == 2.0
        assert thr.command(6.0) == 5.0
        
        # Test MIB
        thr_mib = Thruster(max_thrust=1.0, min_impulse_bit=0.5) 
        # dt=1.0, cmd=0.4 -> Impulse=0.4 < 0.5 -> Should be 0
        assert thr_mib.command(0.4, dt=1.0) == 0.0
        # dt=1.0, cmd=0.6 -> Impulse=0.6 > 0.5 -> Should be 0.6
        assert thr_mib.command(0.6, dt=1.0) == 0.6
        
        chem = ChemicalThruster(max_thrust=100.0, min_on_time=0.01) # MIB = 1.0 Ns
        assert chem.command(150.0) == 100.0
        
        # Test PWM / Min On Time
        # Max=100, MinOn=0.01. Min Impulse = 1.0.
        # dt=0.1s. 
        # Cmd = 5N. Avg Impulse = 0.5Ns < 1.0Ns. Should fail (0).
        assert chem.command(5.0, dt=0.1) == 0.0
        
        # Cmd = 15N. Avg Impulse = 1.5Ns > 1.0Ns. Should pass.
        assert chem.command(15.0, dt=0.1) == 15.0

    def test_electric_thruster(self):
        ethr = ElectricThruster(max_thrust=0.1, isp=1500, power_efficiency=0.5)
        # Power = T * (Isp*g0) / (2 * eta)
        # T=0.1, Isp=1500, g0~9.8, eta=0.5
        # P = 0.1 * 14700 / 1.0 = 1470 W
        p = ethr.get_power_consumption(0.1)
        assert 1400 < p < 1500
