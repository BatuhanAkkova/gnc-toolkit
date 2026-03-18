import unittest
import numpy as np
from gnc_toolkit.edl import (
    ballistic_entry_dynamics,
    lifting_entry_dynamics,
    sutton_grave_heating,
    calculate_g_load
)

class TestEDL(unittest.TestCase):
    def test_ballistic_entry(self):
        # Entry at 120 km
        r = np.array([6491000.0, 0.0, 0.0])
        v = np.array([0.0, 7500.0, 0.0])
        state = np.concatenate([r, v])
        
        cd = 2.0
        area = 1.0 # m^2
        mass = 100.0 # kg
        
        # Derivative check
        deriv = ballistic_entry_dynamics(0.0, state, cd, area, mass)
        
        self.assertEqual(len(deriv), 6)
        # Velocity matches state
        np.testing.assert_allclose(deriv[:3], v)
        
        # Acceleration should include drag (opposite to v) and gravity (opposite to r)
        v_unit = v / np.linalg.norm(v)
        r_unit = r / np.linalg.norm(r)
        
        # Drag is along Y, Gravity is along X
        self.assertLess(deriv[3], 0) # Gravity
        self.assertLess(deriv[4], 0) # Drag

    def test_lifting_entry(self):
        r = np.array([6491000.0, 0.0, 0.0])
        v = np.array([0.0, 7500.0, 0.0])
        state = np.concatenate([r, v])
        
        cl = 0.5
        cd = 1.0
        bank = 0.0
        area = 2.0
        mass = 200.0
        
        deriv_lift = lifting_entry_dynamics(0.0, state, cl, cd, bank, area, mass)
        deriv_ballistic = ballistic_entry_dynamics(0.0, state, cd, area, mass)
        
        # Lift should be perpendicular to velocity
        # v is Y, r is X. h = cross(X, Y) = Z.
        # Vertical lift = cross(Y, Z) = X.
        # So lift should be along X (radial).
        # Gravity is -X, Lift is +X.
        
        self.assertGreater(deriv_lift[3], deriv_ballistic[3])

    def test_heating(self):
        rho = 1e-4
        v = 7500.0
        rn = 1.0
        q = sutton_grave_heating(rho, v, rn)
        
        # q = 1.74153e-4 * sqrt(1e-4) * 7500^3
        # q = 1.74153e-4 * 1e-2 * 421875000000
        # q = 1.74153 * 1e-6 * 421875000000
        # q approx 1.74153 * 421875 approx 734700 W/m^2
        self.assertGreater(q, 100000)

    def test_g_load(self):
        acc = np.array([0.0, 0.0, 9.80665 * 5])
        g = calculate_g_load(acc)
        self.assertAlmostEqual(g, 5.0)

if __name__ == '__main__':
    unittest.main()
