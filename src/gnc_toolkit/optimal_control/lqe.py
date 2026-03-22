"""
Linear Quadratic Estimator (LQE) / Kalman Filter.
"""

import numpy as np
from scipy.linalg import solve_continuous_are

class LQE:
    """
    Linear Quadratic Estimator (LQE) / Kalman Filter.
    
    Designs an optimal observer gain L for the system:
    x_dot = Ax + Bu + Gw
    y = Cx + v
    
    Where:
    w is N(0, Q) (Process Noise)
    v is N(0, R) (Measurement Noise)
    
    The observer dynamics are:
    x_hat_dot = A x_hat + B u + L(y - C x_hat)
    """
    def __init__(self, A, G, C, Q, R):
        """
        Initialize the LQE.
        
        Args:
            A (np.ndarray): State matrix
            G (np.ndarray): Process noise input matrix (often Identity)
            C (np.ndarray): Output matrix
            Q (np.ndarray): Process noise covariance matrix
            R (np.ndarray): Measurement noise covariance matrix
        """
        self.A = np.array(A)
        self.G = np.array(G)
        self.C = np.array(C)
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.P = None
        self.L = None

    def solve(self):
        """
        Solve the Continuous Algebraic Riccati Equation (CARE) for estimation.        
        Mapping:
        X -> P
        a -> A.T
        b -> C.T
        q -> G Q G.T
        r -> R
        """
        a = self.A.T
        b = self.C.T
        q = self.G @ self.Q @ self.G.T
        r = self.R
        
        self.P = solve_continuous_are(a, b, q, r)
        return self.P

    def compute_gain(self):
        """
        Compute the observer gain matrix L.
        L = P C.T inv(R)
        """
        if self.P is None:
            self.solve()
            
        # For robustness, use R instead of inv(R) since R is square
        term = np.linalg.solve(self.R, self.C @ self.P)
        self.L = term.T
        
        return self.L
