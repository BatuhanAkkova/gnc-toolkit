import unittest
import numpy as np
from gnc_toolkit.guidance.porkchop import generate_porkchop_grid

class TestPorkchop(unittest.TestCase):
    def test_generate_porkchop_grid(self):
        # Mock functions for circular orbits
        mu = 398600.4418
        r1 = 7000.0
        r2 = 10000.0
        
        def r_dep_func(t):
            # Planet 1 moves in circle in XY plane
            omega = np.sqrt(mu / r1**3)
            return r1 * np.array([np.cos(omega * t), np.sin(omega * t), 0.0])
            
        def v_dep_func(t):
            omega = np.sqrt(mu / r1**3)
            return r1 * omega * np.array([-np.sin(omega * t), np.cos(omega * t), 0.0])
            
        def r_arr_func(t):
            # Planet 2 moves in circle
            omega = np.sqrt(mu / r2**3)
            return r2 * np.array([np.cos(omega * t), np.sin(omega * t), 0.0])
            
        def v_arr_func(t):
            omega = np.sqrt(mu / r2**3)
            return r2 * omega * np.array([-np.sin(omega * t), np.cos(omega * t), 0.0])

        departure_dates = np.linspace(0, 1000, 5)
        arrival_dates = np.linspace(2000, 3000, 5)
        
        grid = generate_porkchop_grid(departure_dates, arrival_dates, r_dep_func, v_dep_func, r_arr_func, v_arr_func, mu)
        
        self.assertEqual(grid['c3'].shape, (5, 5))
        self.assertEqual(grid['v_inf_arr'].shape, (5, 5))
        
        # Check that TOF is positive
        self.assertTrue(np.all(grid['tof'][~np.isnan(grid['tof'])] > 0))
        
        # Check that some values are computed (not all NaN)
        self.assertFalse(np.all(np.isnan(grid['c3'])))

if __name__ == '__main__':
    unittest.main()
