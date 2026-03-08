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

class TestLQG(unittest.TestCase):
    def test_lqg_stability(self):
        from gnc_toolkit.optimal_control.lqg import LQG
        # Simple 1D system: x_dot = u + w, y = x + v
        A = np.array([[1.0]]) # Unstable system
        B = np.array([[1.0]])
        C = np.array([[1.0]])
        
        Q_lqr = np.array([[1.0]])
        R_lqr = np.array([[1.0]])
        Q_lqe = np.array([[1.0]])
        R_lqe = np.array([[1.0]])
        
        lqg = LQG(A, B, C, Q_lqr, R_lqr, Q_lqe, R_lqe)
        
        # Check gains exist
        self.assertIsNotNone(lqg.K)
        self.assertIsNotNone(lqg.L)
        
        # Initial state and estimate
        x = np.array([1.0])
        lqg.x_hat = np.array([0.0])
        
        # Simulate a few steps
        dt = 0.01
        u = np.array([0.0])
        y = C @ x
        for _ in range(500):
            u = lqg.compute_control(y=y, dt=dt, u_last=u)
            x_dot = A @ x + B @ u
            x = x + x_dot * dt
            y = C @ x
            
        # Should converge towards zero
        self.assertLess(np.abs(x[0]), 1.0)
        self.assertLess(np.abs(x[0] - lqg.x_hat[0]), 0.1)

class TestFiniteHorizonLQR(unittest.TestCase):
    def test_finite_horizon_lqr(self):
        from gnc_toolkit.optimal_control.finite_horizon_lqr import FiniteHorizonLQR
        # Double integrator
        def A_fn(t): return np.array([[0, 1], [0, 0]])
        def B_fn(t): return np.array([[0], [1]])
        def Q_fn(t): return np.eye(2)
        def R_fn(t): return np.eye(1)
        Pf = np.eye(2) * 10.0
        T = 2.0
        
        fhlqr = FiniteHorizonLQR(A_fn, B_fn, Q_fn, R_fn, Pf, T)
        t_span, P_traj = fhlqr.solve(num_points=10)
        
        self.assertEqual(len(t_span), 10)
        self.assertEqual(P_traj.shape, (10, 2, 2))
        
        # Check gain at t=0
        K0 = fhlqr.get_gain(0.0)
        self.assertEqual(K0.shape, (1, 2))
        
        # Gain at end should be R^-1 * B.T * Pf = [[0, 1]] * 10 = [[0, 10]]
        KT = fhlqr.get_gain(2.0)
        np.testing.assert_allclose(KT, [[0.0, 10.0]], atol=1e-5)

class TestHInfinity(unittest.TestCase):
    def test_h_infinity_scalar(self):
        from gnc_toolkit.optimal_control.h_infinity import HInfinityController
        # x_dot = x + w + u
        # z = [x; u]
        A = np.array([[1.0]])
        B1 = np.array([[1.0]])
        B2 = np.array([[1.0]])
        Q = np.array([[1.0]])
        R = np.array([[1.0]])
        gamma = 2.0
        
        hinf = HInfinityController(A, B1, B2, Q, R, gamma)
        K = hinf.compute_gain()
        
        # Check stability of A - B2*K
        self.assertLess(A[0,0] - B2[0,0] * K[0,0], 0)
        
    def test_h_infinity_fail(self):
        from gnc_toolkit.optimal_control.h_infinity import HInfinityController
        A = np.array([[1.0]])
        B1 = np.array([[10.0]]) # Huge disturbance
        B2 = np.array([[1.0]])
        Q = np.array([[1.0]])
        R = np.array([[1.0]])
        gamma = 0.1 # Too strict attenuation for large B1
        
        hinf = HInfinityController(A, B1, B2, Q, R, gamma)
        with self.assertRaises(ValueError):
            hinf.solve()

if __name__ == '__main__':
    unittest.main()