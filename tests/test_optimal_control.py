import sys
import os
import unittest
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from gnc_toolkit.optimal_control.lqr import LQR

class TestLQR(unittest.TestCase):
    def test_double_integrator(self):
        # Double integrator system
        # x_dot = v
        # v_dot = u
        A = np.array([[0, 1], [0, 0]])
        B = np.array([[0], [1]])
        Q = np.eye(2)
        R = np.eye(1)
        
        lqr = LQR(A, B, Q, R)
        K = lqr.compute_gain()
        
        # Check dimensions
        self.assertEqual(K.shape, (1, 2))
        
        # Check closed loop stability
        # A_cl = A - B*K
        A_cl = A - B @ K
        eigenvalues = np.linalg.eigvals(A_cl)
        
        # All eigenvalues should have negative real parts
        for eig in eigenvalues:
            self.assertLess(eig.real, 0)
            
        print(f"Computed LQR Gain: {K}")
        print(f"Closed Loop Eigenvalues: {eigenvalues}")

    def test_lqe_scalar(self):
        # Scalar system check
        # x_dot = -x + w
        # y = x + v
        from gnc_toolkit.optimal_control.lqe import LQE
        
        A = [[-1]]
        G = [[1]]
        C = [[1]]
        Q = [[1]]
        R = [[1]]
        
        lqe = LQE(A, G, C, Q, R)
        P = lqe.solve()
        L = lqe.compute_gain()
        
        # Theoretical P: P^2 + 2P - 1 = 0 => P = sqrt(2) - 1
        expected_P = np.sqrt(2) - 1
        self.assertAlmostEqual(P[0,0], expected_P, places=5)
        
        # Theoretical L: P * 1 * 1 = P
        self.assertAlmostEqual(L[0,0], expected_P, places=5)
        
        print(f"Computed LQE Covariance P: {P}")
        print(f"Computed LQE Gain L: {L}")

    def test_sliding_mode(self):
        # dot_x = u
        # s = x
        # stable if K > 0
        from gnc_toolkit.optimal_control.sliding_mode import SlidingModeController
        
        surface_func = lambda x, t: x[0]
        K = 1.0
        smc = SlidingModeController(surface_func, K, chattering_reduction=False)
        
        # Test positive state
        x_pos = np.array([2.0])
        u_pos = smc.compute_control(x_pos)
        self.assertEqual(u_pos, -1.0)
        
        # Test negative state
        x_neg = np.array([-2.0])
        u_neg = smc.compute_control(x_neg)
        self.assertEqual(u_neg, 1.0)
        
        # Test saturation
        smc_sat = SlidingModeController(surface_func, K, chattering_reduction=True, boundary_layer=1.0)
        # s = 0.5, phi = 1.0 => s/phi = 0.5 => u = -K * 0.5 = -0.5
        x_iny = np.array([0.5])
        u_sat = smc_sat.compute_control(x_iny)
        self.assertEqual(u_sat, -0.5)

    def test_mpc(self):
        from gnc_toolkit.optimal_control.mpc import LinearMPC
        
        # Discrete Double Integrator (dt=0.1)
        dt = 0.1
        A = np.array([[1, dt], [0, 1]])
        B = np.array([[0], [dt]])
        Q = np.diag([1.0, 0.1])
        R = np.eye(1) * 0.01
        
        mpc = LinearMPC(A, B, Q, R, horizon=10, u_min=[-1.0], u_max=np.array([1.0]), x_max=np.array([1.5, 10.0]))
        
        x0 = np.array([1.0, 0.0]) # Initial pos=1, vel=0
        
        # Solve
        U = mpc.solve(x0)
        
        # Check output shape
        self.assertEqual(U.shape, (10, 1))
        
        # Check first control action is negative (pushing back to 0)
        self.assertLess(U[0,0], 0)
        
        # Check constraints
        self.assertTrue(np.all(U >= -1.0))
        self.assertTrue(np.all(U <= 1.0))
        
        # Verify state constraints
        # Propagate manually to check state
        x_curr = x0.copy()
        for i in range(10):
            x_curr = A @ x_curr + B @ U[i]
        # Check Position max constraint (x[0] <= 1.5)
            self.assertLessEqual(x_curr[0], 1.5001) # Add small tol for solver
            
        print(f"MPC Control Sequence: {U.flatten()}")

    def test_nonlinear_mpc(self):
        from gnc_toolkit.optimal_control.mpc import NonlinearMPC
        
        # Nonlinear system: Simple Pendulum
        # x1_dot = x2
        # x2_dot = -sin(x1) + u
        dt = 0.1
        
        def dynamics(x, u):
            x1, x2 = x
            x1_next = x1 + x2 * dt
            x2_next = x2 + (-np.sin(x1) + u[0]) * dt
            return np.array([x1_next, x2_next])
            
        def cost(x, u):
            # Regulate to 0
            return (x[0]**2 + x[1]**2) + 0.1 * u[0]**2
            
        def terminal_cost(x):
            return 10.0 * (x[0]**2 + x[1]**2)
            
        nmpc = NonlinearMPC(dynamics_func=dynamics, 
                            cost_func=cost, 
                            terminal_cost_func=terminal_cost,
                            horizon=10, 
                            nx=2, nu=1,
                            u_min=[-2.0], u_max=[2.0])
                            
        x0 = np.array([np.pi/2, 0.0]) # Start at 90 degrees
        
        U = nmpc.solve(x0)
        
        self.assertEqual(U.shape, (10, 1))
        # Check constraints
        self.assertTrue(np.all(U >= -2.001))
        self.assertTrue(np.all(U <= 2.001))
        
        print(f"NMPC Control Sequence: {U.flatten()}")

    def test_feedback_linearization(self):
        from gnc_toolkit.optimal_control.feedback_linearization import FeedbackLinearization
        
        # System: dot_x = x^2 + u
        # f(x) = x^2
        # g(x) = 1
        
        f_func = lambda x: x**2
        g_func = lambda x: 1.0
        
        fl_controller = FeedbackLinearization(f_func, g_func)
        
        x = np.array([2.0])
        # Desired: dot_x = -x = -2.0
        v = -x
        
        # u = (v - f(x))/g(x) = (-2 - 4) / 1 = -6
        u = fl_controller.compute_control(x, v)
        
        self.assertEqual(u, -6.0)

if __name__ == '__main__':
    unittest.main()