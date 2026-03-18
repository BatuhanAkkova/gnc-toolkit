import unittest
import numpy as np
from gnc_toolkit.guidance.continuous_thrust import (
    q_law_guidance,
    zem_zev_guidance,
    gravity_turn_guidance,
    apollo_dps_guidance
)

class TestContinuousThrust(unittest.TestCase):
    def test_q_law(self):
        # LEO state
        r = np.array([7000000.0, 0.0, 0.0])
        v = np.array([0.0, 7550.0, 0.0])
        mu = 398600.4415e9
        
        # Target a slightly higher orbit (a = 7500 km)
        target_a = 7500000.0
        target_oe = np.array([target_a, 0.0, 0.0, 0.0, 0.0])
        
        f_max = 0.01 # m/s^2
        
        f_eci = q_law_guidance(r, v, target_oe, mu, f_max)
        
        self.assertEqual(len(f_eci), 3)
        self.assertAlmostEqual(np.linalg.norm(f_eci), f_max, delta=1e-10)
        
        # For a simple orbit raising, thrust should be primarily tangential
        # ECI velocity is [0, 7550, 0], so tangential is along Y
        self.assertGreater(f_eci[1], 0)

    def test_zem_zev(self):
        r = np.array([1000.0, 0.0, 0.0])
        v = np.array([-10.0, 0.0, 0.0])
        r_target = np.array([0.0, 0.0, 0.0])
        v_target = np.array([0.0, 0.0, 0.0])
        t_go = 100.0
        
        # Expected ZEM = 0 - (1000 + (-10)*100) = 0
        # Expected ZEV = 0 - (-10) = 10
        # a_cmd = (6/100^2)*0 - (2/100)*10 = -0.2
        
        a_cmd = zem_zev_guidance(r, v, r_target, v_target, t_go)
        np.testing.assert_allclose(a_cmd, np.array([-0.2, 0.0, 0.0]), atol=1e-5)

    def test_gravity_turn(self):
        v = np.array([0.0, 0.0, -100.0])
        f_mag = 20.0
        
        # Descent mode: thrust should be opposite to velocity (upwards)
        a_thrust = gravity_turn_guidance(v, f_mag, mode='descent')
        np.testing.assert_allclose(a_thrust, np.array([0.0, 0.0, 20.0]))
        
        # Ascent mode: thrust should be along velocity
        a_thrust = gravity_turn_guidance(v, f_mag, mode='ascent')
        np.testing.assert_allclose(a_thrust, np.array([0.0, 0.0, -20.0]))

    def test_apollo_dps(self):
        t_go = 50.0
        r = np.array([500.0, 0.0, 0.0])
        v = np.array([-20.0, 0.0, 0.0])
        r_target = np.array([0.0, 0.0, 0.0])
        v_target = np.array([0.0, 0.0, 0.0])
        a_target = np.array([0.0, 0.0, 0.0])
        
        a_cmd = apollo_dps_guidance(t_go, r, v, r_target, v_target, a_target)
        
        # delta_r = 0 - (500 - 20*50) = 500
        # delta_v = 0 - (-20) = 20
        # a_cmd = 0 + (12/2500)*500 + (6/50)*20 = 2.4 + 2.4 = 4.8
        
        self.assertAlmostEqual(a_cmd[0], 4.8, delta=1e-5)

if __name__ == '__main__':
    unittest.main()
