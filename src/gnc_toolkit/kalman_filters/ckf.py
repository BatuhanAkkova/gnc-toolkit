import numpy as np
from scipy.linalg import cholesky

class CKF:
    """
    Cubature Kalman Filter (CKF).
    Based on the spherical-radial cubature rule.
    Alternative to UKF with fixed weights and potentially better stability for high dimensions.
    """
    def __init__(self, dim_x, dim_z):
        """
        Initialize the CKF.
        dim_x: Dimension of the state vector
        dim_z: Dimension of the measurement vector
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        # CKF uses 2n cubature points
        self.m = 2 * dim_x
        self.weights = 1.0 / self.m
        
        # Cubature points set (unit vectors along each dimension)
        self.xi = np.zeros((dim_x, self.m))
        for i in range(dim_x):
            self.xi[i, i] = np.sqrt(dim_x)
            self.xi[i, i + dim_x] = -np.sqrt(dim_x)
            
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)

    def predict(self, dt, fx, Q=None, **kwargs):
        """
        Predict step.
        dt: Time step
        fx: State transition function f(x, dt, **kwargs) -> x_new
        Q: Optional process noise covariance
        """
        if Q is None: Q = self.Q
        
        # Generate cubature points from predicted distribution
        points = self._generate_cubature_points(self.x, self.P)
        
        # Propagate cubature points
        points_f = np.zeros((self.m, self.dim_x))
        for i in range(self.m):
            points_f[i] = fx(points[:, i], dt, **kwargs)
            
        # Predicted mean
        self.x = self.weights * np.sum(points_f, axis=0)
        
        # Predicted covariance
        self.P = np.zeros((self.dim_x, self.dim_x))
        for i in range(self.m):
            dx = points_f[i] - self.x
            self.P += self.weights * np.outer(dx, dx)
        self.P += Q

    def update(self, z, hx, R=None, **kwargs):
        """
        Update step.
        z: Measurement vector
        hx: Measurement function h(x, **kwargs) -> z_pred
        R: Optional measurement noise covariance
        """
        if R is None: R = self.R
        
        # Regenerate cubature points from predicted distribution
        points_f = self._generate_cubature_points(self.x, self.P)
        
        # Transform to measurement space
        points_h = np.zeros((self.m, self.dim_z))
        for i in range(self.m):
            points_h[i] = hx(points_f[:, i], **kwargs)
            
        # Mean measurement
        zp = self.weights * np.sum(points_h, axis=0)
        
        # Innovation covariance S and Cross-covariance Pxz
        S = np.zeros((self.dim_z, self.dim_z))
        Pxz = np.zeros((self.dim_x, self.dim_z))
        
        for i in range(self.m):
            dx = points_f[:, i] - self.x
            dz = points_h[i] - zp
            S += self.weights * np.outer(dz, dz)
            Pxz += self.weights * np.outer(dx, dz)
            
        S += R
        
        # Kalman gain
        K = np.dot(Pxz, np.linalg.inv(S))
        
        # Correct mean and covariance
        self.x += np.dot(K, z - zp)
        self.P -= np.dot(K, np.dot(S, K.T))

    def _generate_cubature_points(self, x, P):
        """Generate 2n cubature points using Cholesky factor of P."""
        # Ensure symmetry
        P_sym = (P + P.T) / 2 + np.eye(self.dim_x) * 1e-12
        try:
            S = cholesky(P_sym, lower=True)
            return x[:, np.newaxis] + np.dot(S, self.xi)
        except np.linalg.LinAlgError:
            # Fallback if P is not PSD
            from scipy.linalg import sqrtm
            S = sqrtm(P_sym).real
            return x[:, np.newaxis] + np.dot(S, self.xi)
