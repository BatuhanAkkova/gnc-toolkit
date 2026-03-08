import unittest
import numpy as np
from gnc_toolkit.guidance.maneuvers import (
    delta_v_budget,
    raan_change,
    optimal_combined_maneuver,
    hohmann_transfer
)

class TestManeuversExtended(unittest.TestCase):
    def test_delta_v_budget(self):
        initial_mass = 1000.0 # kg
        dv = 0.5 # km/s
        isp = 300.0 # s
        
        m_prop = delta_v_budget(initial_mass, dv, isp)
        
        # Hand calculation check
        # g0 = 0.00980665 km/s^2
        # m_prop = 1000 * (1 - exp(-0.5 / (300 * 0.00980665)))
        # m_prop approx 1000 * (1 - exp(-0.1699)) approx 1000 * (1 - 0.8437) = 156.3 kg
        self.assertAlmostEqual(m_prop, 156.3, delta=1.0)
        self.assertGreater(m_prop, 0)
        self.assertLess(m_prop, initial_mass)

    def test_raan_change(self):
        v = 7.5 # km/s
        i = np.radians(28.5)
        delta_raan = np.radians(1.0)
        
        dv = raan_change(v, i, delta_raan)
        
        # For small delta_raan, dv approx v * sin(i) * delta_raan
        # 7.5 * sin(28.5) * (1 * pi/180) approx 7.5 * 0.477 * 0.01745 approx 0.0624 km/s
        self.assertAlmostEqual(dv, 0.0624, delta=0.01)

    def test_optimal_combined_maneuver(self):
        r1 = 7000.0
        r2 = 42164.0
        delta_i = np.radians(28.5)
        mu = 398600.4418
        
        dv_opt, di1, di2 = optimal_combined_maneuver(r1, r2, delta_i, mu)
        
        # Check that sum of di is delta_i
        self.assertAlmostEqual(di1 + di2, delta_i)
        
        # Standard Hohmann + simple plane change at r2
        dv1_h, dv2_h, _ = hohmann_transfer(r1, r2, mu)
        v_c2 = np.sqrt(mu / r2)
        # Plane change at r2: 2 * v_c2 * sin(delta_i/2)
        dv_pc = 2 * v_c2 * np.sin(delta_i / 2.0)
        total_non_opt = dv1_h + dv2_h + dv_pc
        
        # Optimal should be better
        self.assertLess(dv_opt, total_non_opt)
        
        # Also check against combined_plane_change formula if we happen to do all PC at r2
        from gnc_toolkit.guidance.maneuvers import combined_plane_change
        # v_trans_a = velocity at apoapsis of transfer orbit
        a_trans = (r1 + r2) / 2.0
        v_trans_a = np.sqrt(mu * (2/r2 - 1/a_trans))
        dv2_combined = combined_plane_change(v_trans_a, v_c2, delta_i)
        total_combined_at_r2 = dv1_h + dv2_combined
        
        self.assertLessEqual(dv_opt, total_combined_at_r2 + 1e-6)

if __name__ == '__main__':
    unittest.main()
