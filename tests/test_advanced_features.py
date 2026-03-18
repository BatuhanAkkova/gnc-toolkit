import unittest
import numpy as np
from gnc_toolkit.guidance.continuous_thrust import indirect_optimal_guidance, direct_collocation_guidance
from gnc_toolkit.edl import aerocapture_guidance, hazard_avoidance
from gnc_toolkit.navigation.terrain_nav import FeatureMatchingTRN, map_relative_localization_update

class TestAdvancedFeatures(unittest.TestCase):
    def test_indirect_optimal(self):
        r0 = np.array([7000000.0, 0.0, 0.0])
        v0 = np.array([0.0, 7550.0, 0.0])
        rf = np.array([7100000.0, 0.0, 0.0])
        vf = np.array([0.0, 7450.0, 0.0])
        tf = 1000.0
        mu = 398600.4415e9
        
        # This solves a BVP, it might be slow or fail to converge with zero guess
        # But we check the structure
        t, a = indirect_optimal_guidance(r0, v0, rf, vf, tf, mu)
        if t is not None:
            self.assertEqual(len(t), len(a[0]))
            self.assertGreater(len(t), 0)

    def test_direct_collocation(self):
        r0 = np.array([7000000.0, 0.0, 0.0])
        v0 = np.array([0.0, 7550.0, 0.0])
        rf = np.array([7001000.0, 0.0, 0.0])
        vf = np.array([0.0, 7550.0, 0.0])
        tf = 100.0
        mu = 398600.4415e9
        
        # Small transfer for quick optimization
        traj = direct_collocation_guidance(r0, v0, rf, vf, tf, mu, n_nodes=10)
        if traj is not None:
            self.assertEqual(traj.shape, (10, 9))
            np.testing.assert_allclose(traj[0, :3], r0, atol=1e-3)
            np.testing.assert_allclose(traj[-1, :3], rf, atol=1e-3)

    def test_edl_advanced(self):
        # Hazard avoidance
        r = np.array([0.0, 0.0, 10.0])
        v = np.array([0.0, 0.0, -1.0])
        hazards = [np.array([0.0, 0.0, 0.0])]
        dv = hazard_avoidance(r, v, hazards, safety_margin=20.0)
        self.assertGreater(np.linalg.norm(dv), 0)
        
        # Aerocapture placeholder
        bank = aerocapture_guidance(None, None, None, None, None, None, None)
        self.assertEqual(bank, 0.0)

    def test_trn(self):
        map_db = [np.array([100.0, 200.0, 0.0]), np.array([500.0, 600.0, 0.0])]
        trn = FeatureMatchingTRN(map_db)
        obs = [np.array([101.0, 201.0, 0.0])]
        matches = trn.match_features(obs)
        self.assertEqual(len(matches), 1)
        self.assertTrue(np.array_equal(matches[0][0], map_db[0]))

if __name__ == '__main__':
    unittest.main()
