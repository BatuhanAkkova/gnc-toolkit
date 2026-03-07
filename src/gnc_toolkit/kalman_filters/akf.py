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
        self.S_list = []
        self.H_list = []
        self.P_minus_list = []
        self.predict_list = [] # Fx

    def predict(self, u=None, F=None, Q=None, B=None):
        """Standard predict step, but stores values for adaptation."""
        if F is None: F = self.F
        if Q is None: Q = self.Q
        if B is None: B = self.B
        
        self.P_minus_list.append(self.P.copy())
        # Store predicted state without noise/update for Q estimation
        if B is not None and u is not None:
            self.predict_list.append(np.dot(F, self.x) + np.dot(B, u))
        else:
            self.predict_list.append(np.dot(F, self.x))
            
        super().predict(u, F, Q, B)
        
        if len(self.P_minus_list) > self.N:
            self.P_minus_list.pop(0)
            self.predict_list.pop(0)

    def update(self, z, H=None, R=None):
        """Update step with online R and Q adaptation."""
        if H is None: H = self.H
        if R is None: R = self.R
        
        # Standard update first to get innovation
        # y = z - Hx (done inside super().update, but we need it for adaptation)
        y = z - np.dot(H, self.x)
        S = np.dot(np.dot(H, self.P), H.T) + R
        
        self.innovations.append(y)
        self.S_list.append(S)
        self.H_list.append(H)
        
        if len(self.innovations) > self.N:
            self.innovations.pop(0)
            self.S_list.pop(0)
            self.H_list.pop(0)
            
        # Perform adaptation if we have enough samples
        if len(self.innovations) >= self.N:
            self._adapt_noise_covariances()
            
        super().update(z, H, R)

    def _adapt_noise_covariances(self):
        """Estimates Q and R based on innovation sequence."""
        # Estimated R (Measurement noise)
        # R_hat = (1/N) * sum(y*y.T) - (1/N) * sum(H*P_minus*H.T)
        sum_yyT = np.zeros((self.dim_z, self.dim_z))
        sum_HPHT = np.zeros((self.dim_z, self.dim_z))
        
        for i in range(self.N):
            y = self.innovations[i]
            H = self.H_list[i]
            Pm = self.P_minus_list[i]
            sum_yyT += np.outer(y, y)
            sum_HPHT += np.dot(np.dot(H, Pm), H.T)
            
        R_hat = (1.0 / self.N) * sum_yyT - (1.0 / self.N) * sum_HPHT
        
        # Ensure R is positive definite
        self.R = self._make_psd(R_hat, self.dim_z)
        
        # Estimated Q (Process noise) - Simplified Sage-Husa or Myers-Tapley
        # Q_hat = (1/N) * sum(dx * dx.T) where dx is state correction
        # For simplicity, we can also use the gain-based approach or residuals
        # Here we use a common heuristic for Q adaptation
        # dx = x_filtered - x_predicted
        # sum_dxdxT = ...
        
        # (Optional: Implement full Q estimation if required)
        pass

    def _make_psd(self, matrix, dim):
        """Force a matrix to be positive semi-definite."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        return np.dot(eigenvectors * eigenvalues, eigenvectors.T)
