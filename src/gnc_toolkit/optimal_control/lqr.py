"""
Linear Quadratic Regulator (LQR) Controller.
"""

import numpy as np
from scipy.linalg import solve_continuous_are


class LQR:
    """
    Linear Quadratic Regulator (LQR) Controller.

    Minimizes the cost function:
    J = integral(x.T * Q * x + u.T * R * u) dt

    The control law is u = -K * x
    """

    def __init__(self, A, B, Q, R):
        """
        Initialize the LQR controller.

        Args:
            A (np.ndarray): State matrix
            B (np.ndarray): Input matrix
            Q (np.ndarray): State cost matrix
            R (np.ndarray): Input cost matrix
        """
        self.A = np.array(A)
        self.B = np.array(B)
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.P = None
        self.K = None

    def solve(self):
        """
        Solve the Continuous Algebraic Riccati Equation (CARE).
        P * A + A.T * P - P * B * inv(R) * B.T * P + Q = 0
        """
        self.P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        return self.P

    def compute_gain(self):
        """
        Compute the feedback gain matrix K.
        K = inv(R) * B.T * P
        """
        if self.P is None:
            self.solve()

        # For robustness, use R instead of inv(R) since R is square
        self.K = np.linalg.solve(self.R, self.B.T @ self.P)
        return self.K
