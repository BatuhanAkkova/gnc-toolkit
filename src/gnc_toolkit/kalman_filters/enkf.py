"""
Ensemble Kalman Filter (EnKF) using Monte Carlo samples for covariance representation.
"""

import numpy as np

class EnKF:
    """
    Ensemble Kalman Filter (EnKF).
    Uses an ensemble of states to represent the covariance matrix.
    Efficient for high-dimensional systems where the full covariance is too large.
    """
    def __init__(self, dim_x, dim_z, ensemble_size=50):
        """
        Initialize the EnKF.
        dim_x: Dimension of the state vector
        dim_z: Dimension of the measurement vector
        ensemble_size: Number of ensemble members (N)
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.N = ensemble_size
        
        # Ensemble of states: shape (dim_x, N)
        self.X = np.zeros((dim_x, self.N))
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)
        
    def initialize_ensemble(self, x_mean, P):
        """
        Initialize the ensemble around a mean with covariance P.
        """
        self.X = np.random.multivariate_normal(x_mean, P, self.N).T

    def predict(self, dt, fx, Q=None, **kwargs):
        """
        Predict step.
        dt: Time step
        fx: State transition function f(x, dt, **kwargs) -> x_new
        Q: Optional process noise covariance
        """
        if Q is None: Q = self.Q
        
        # Propagate each ensemble member
        for i in range(self.N):
            # Propagate through nonlinear model
            self.X[:, i] = fx(self.X[:, i], dt, **kwargs)
            
            # Add process noise to each member
            noise = np.random.multivariate_normal(np.zeros(self.dim_x), Q)
            self.X[:, i] += noise

    def update(self, z, hx, R=None, **kwargs):
        """
        Update step.
        z: Measurement vector
        hx: Measurement function h(x, **kwargs) -> z_pred
        R: Optional measurement noise covariance
        """
        if R is None: R = self.R
        
        # Transform ensemble to measurement space
        Z_ensemble = np.zeros((self.dim_z, self.N))
        for i in range(self.N):
            Z_ensemble[:, i] = hx(self.X[:, i], **kwargs)
            
        # Sample mean of measurement ensemble
        z_mean = np.mean(Z_ensemble, axis=1, keepdims=True)
        
        # Calculate ensemble anomalies (perturbations)
        X_mean = np.mean(self.X, axis=1, keepdims=True)
        A = self.X - X_mean  # State anomalies
        B = Z_ensemble - z_mean  # Measurement anomalies
        
        # Perturbed measurements (adding noise for each ensemble member)
        Z_perturbed = np.zeros((self.dim_z, self.N))
        for i in range(self.N):
            noise = np.random.multivariate_normal(np.zeros(self.dim_z), R)
            Z_perturbed[:, i] = z + noise
            
        # Innovation
        D = Z_perturbed - Z_ensemble
        
        # Innovation covariance S = (1/(N-1)) * B * B.T + R
        S = (1.0 / (self.N - 1)) * np.dot(B, B.T) + R
        
        # Cross-covariance Pxz = (1/(N-1)) * A * B.T
        Pxz = (1.0 / (self.N - 1)) * np.dot(A, B.T)
        
        # Kalman Gain K = Pxz * inv(S)
        K = np.dot(Pxz, np.linalg.inv(S))
        
        # Correct each ensemble member
        self.X += np.dot(K, D)

    @property
    def x(self):
        """Returns the ensemble mean state."""
        return np.mean(self.X, axis=1)

    @property
    def P(self):
        """Returns the ensemble covariance matrix."""
        A = self.X - np.mean(self.X, axis=1, keepdims=True)
        return (1.0 / (self.N - 1)) * np.dot(A, A.T)
