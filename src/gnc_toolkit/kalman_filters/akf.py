"""
Adaptive Kalman Filter (AKF) with online covariance estimation (Myers-Tapley).
"""

import numpy as np
from gnc_toolkit.kalman_filters.kf import KF

class AKF(KF):
    """
    Adaptive Kalman Filter (AKF).
    Estimates process noise covariance (Q) and measurement noise covariance (R) online
    using the innovation sequence (Myers-Tapley method).
    Inherits from the standard Linear Kalman Filter (KF).
    """
    def __init__(self, dim_x, dim_z, window_size=20):
        """
        Initialize the AKF.
        dim_x: Dimension of the state vector
        dim_z: Dimension of the measurement vector
        window_size: Moving window size for covariance estimation
        """
        super().__init__(dim_x, dim_z)
        self.N = window_size
        self.innovations = []
        self.H_list = []
        self.P_minus_list = []
        self.FPF_T_list = []
        self.dx_list = []
        self.P_plus_list = []

    def predict(self, u=None, F=None, Q=None, B=None):
        """Standard predict step, but stores values for adaptation."""
        if F is None: F = self.F
        if Q is None: Q = self.Q
        if B is None: B = self.B
        
        # Store F * P_{k-1|k-1} * F^T for Q estimation
        self.FPF_T_list.append(np.dot(np.dot(F, self.P), F.T))
        
        super().predict(u, F, Q, B)
        
        # Store P_{k|k-1} for R estimation
        self.P_minus_list.append(self.P.copy())
        
        if len(self.P_minus_list) > self.N:
            self.P_minus_list.pop(0)
            self.FPF_T_list.pop(0)

    def update(self, z, H=None, R=None):
        """Update step with online R and Q adaptation."""
        if H is None: H = self.H
        if R is None: R = self.R
        
        x_minus = self.x.copy()
        y = z - np.dot(H, self.x)
        
        super().update(z, H, R)
        
        dx = self.x - x_minus
        
        self.innovations.append(y)
        self.H_list.append(H)
        self.dx_list.append(dx)
        self.P_plus_list.append(self.P.copy())
        
        if len(self.innovations) > self.N:
            self.innovations.pop(0)
            self.H_list.pop(0)
            self.dx_list.pop(0)
            self.P_plus_list.pop(0)
            
        # Perform adaptation if we have enough samples
        if len(self.innovations) >= self.N:
            self._adapt_noise_covariances()

    def _adapt_noise_covariances(self):
        """Estimates Q and R based on innovation sequence."""
        sum_yyT = np.zeros((self.dim_z, self.dim_z))
        sum_HPHT = np.zeros((self.dim_z, self.dim_z))
        sum_Q = np.zeros((self.dim_x, self.dim_x))
        
        for i in range(self.N):
            y = self.innovations[i]
            H = self.H_list[i]
            Pm = self.P_minus_list[i]
            dx = self.dx_list[i]
            P_plus = self.P_plus_list[i]
            FPF_T = self.FPF_T_list[i]
            
            sum_yyT += np.outer(y, y)
            sum_HPHT += np.dot(np.dot(H, Pm), H.T)
            
            # Myers-Tapley formula for Q
            sum_Q += np.outer(dx, dx) + P_plus - FPF_T
            
        R_hat = (1.0 / self.N) * sum_yyT - (1.0 / self.N) * sum_HPHT
        self.R = self._make_psd(R_hat, self.dim_z)
        
        Q_hat = (1.0 / self.N) * sum_Q
        self.Q = self._make_psd(Q_hat, self.dim_x)

    def _make_psd(self, matrix, dim):
        """Force a matrix to be positive semi-definite."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        return np.dot(eigenvectors * eigenvalues, eigenvectors.T)
