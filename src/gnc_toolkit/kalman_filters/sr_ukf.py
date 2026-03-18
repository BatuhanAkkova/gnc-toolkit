import numpy as np
from scipy.linalg import qr, cholesky, solve_triangular

class SRUKF:
    """
    Square-Root Unscented Kalman Filter (SR-UKF).
    Provides better numerical stability and efficiency than standard UKF
    by propagating the Cholesky factor (S) of the covariance matrix P = S*S.T.
    """
    def __init__(self, dim_x, dim_z, alpha=1e-3, beta=2.0, kappa=0.0):
        """
        Initialize the SR-UKF.
        dim_x: Dimension of the state vector
        dim_z: Dimension of the measurement vector
        alpha, beta, kappa: UKF tuning parameters
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        self.lambda_ = alpha**2 * (dim_x + kappa) - dim_x
        self.gamma = np.sqrt(dim_x + self.lambda_)
        
        self.num_sigmas = 2 * dim_x + 1
        self.Wm = np.zeros(self.num_sigmas)
        self.Wc = np.zeros(self.num_sigmas)
        
        self.Wm[0] = self.lambda_ / (dim_x + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)
        
        weight = 1.0 / (2 * (dim_x + self.lambda_))
        for i in range(1, self.num_sigmas):
            self.Wm[i] = weight
            self.Wc[i] = weight
            
        self.x = np.zeros(dim_x)
        self.S = np.eye(dim_x)  # Cholesky factor of covariance P
        
        # Square-root noise covariances
        self.Qs = np.eye(dim_x)
        self.Rs = np.eye(dim_z)

    def predict(self, dt, fx, Qs=None, **kwargs):
        """
        Predict step.
        dt: Time step
        fx: State transition function f(x, dt, **kwargs) -> x_new
        Qs: Optional square-root process noise covariance (S_Q)
        """
        if Qs is None: Qs = self.Qs
        
        # Generate sigma points using S
        sigmas = self._generate_sigma_points()
        
        # Propagate sigma points
        sigmas_f = np.zeros((self.num_sigmas, self.dim_x))
        for i in range(self.num_sigmas):
            sigmas_f[i] = fx(sigmas[i], dt, **kwargs)
            
        # Predicted mean
        self.x = np.dot(self.Wm, sigmas_f)
        
        # Update square-root covariance S
        # Compute S- using QR decomposition of the propagated sigma points and process noise
        X = np.sqrt(self.Wc[1]) * (sigmas_f[1:] - self.x)
        
        # QR decomposition: Q, R = qr(A), where A is (N + 2L) x L
        # We need the R part of [X.T, Qs.T].T
        _, St = qr(np.vstack((X, Qs)), mode='economic')
        self.S = St[:self.dim_x, :self.dim_x].T
        
        # Cholesky rank-1 update for the first weight (might be negative)
        dx = sigmas_f[0] - self.x
        self.S = self._cholesky_update(self.S, dx, self.Wc[0])

    def update(self, z, hx, Rs=None, **kwargs):
        """
        Update step.
        z: Measurement vector
        hx: Measurement function h(x, **kwargs) -> z_pred
        Rs: Optional square-root measurement noise covariance (S_R)
        """
        if Rs is None: Rs = self.Rs
        
        # Regenerate sigma points
        sigmas_f = self._generate_sigma_points()
        
        # Transform to measurement space
        sigmas_h = np.zeros((self.num_sigmas, self.dim_z))
        for i in range(self.num_sigmas):
            sigmas_h[i] = hx(sigmas_f[i], **kwargs)
            
        # Mean measurement
        zp = np.dot(self.Wm, sigmas_h)
        
        # Square-root innovation covariance Sy
        H = np.sqrt(self.Wc[1]) * (sigmas_h[1:] - zp)
        _, St = qr(np.vstack((H, Rs)), mode='economic')
        Sy = St[:self.dim_z, :self.dim_z].T
        
        # Rank-1 update for first weight
        dz_0 = sigmas_h[0] - zp
        Sy = self._cholesky_update(Sy, dz_0, self.Wc[0])
        
        # Cross-covariance Pxz
        Pxz = np.zeros((self.dim_x, self.dim_z))
        for i in range(self.num_sigmas):
            dx = sigmas_f[i] - self.x
            dz = sigmas_h[i] - zp
            Pxz += self.Wc[i] * np.outer(dx, dz)
            
        # Kalman gain K = (Pxz / Sy.T) / Sy
        # In SRUKF: K = Pxz * inv(Sy.T) * inv(Sy)
        # Using triangular solve for stability
        K = solve_triangular(Sy, solve_triangular(Sy, Pxz.T, lower=True), lower=True, trans='T').T
        
        # Correct mean and square-root covariance
        self.x += np.dot(K, z - zp)
        
        # U = K * Sy
        U = np.dot(K, Sy)
        for i in range(self.dim_z):
            self.S = self._cholesky_update(self.S, U[:, i], -1.0)

    def _generate_sigma_points(self):
        sigmas = np.zeros((self.num_sigmas, self.dim_x))
        sigmas[0] = self.x
        for i in range(self.dim_x):
            sigmas[i+1] = self.x + self.gamma * self.S[:, i]
            sigmas[i+1+self.dim_x] = self.x - self.gamma * self.S[:, i]
        return sigmas

    def _cholesky_update(self, S, v, weight):
        """
        Performs rank-1 Cholesky update or downdate.
        S: Current Cholesky factor (lower triangular)
        v: Vector to update with
        weight: Scalar weight (positive for update, negative for downdate)
        """
        # Note: Scipy doesn't have a direct rank-1 update for Cholesky.        
        # S_new * S_new.T = S*S.T + sigma * v * v.T
        # For positive weights:
        if weight > 0:
            v_scaled = np.sqrt(weight) * v
            # Use QR to get updated S
            _, St = qr(np.vstack((S.T, v_scaled)), mode='economic')
            return St[:S.shape[0], :S.shape[1]].T
        else:
            # Downdate is trickier. For now, let's use the reconstruction for downdate if needed,
            # or avoid it if possible. In SRUKF update, we use downdates.
            # Real implementation would use Givens rotations.
            P = np.dot(S, S.T) + weight * np.outer(v, v)
            try:
                return cholesky(P, lower=True)
            except np.linalg.LinAlgError:
                # Add jitter for stability
                return cholesky(P + np.eye(S.shape[0]) * 1e-12, lower=True)

    @property
    def P(self):
        """Returns the full covariance matrix P = S*S.T"""
        return np.dot(self.S, self.S.T)
