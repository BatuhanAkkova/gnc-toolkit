import numpy as np
from scipy.linalg import cholesky, sqrtm

class UKF:
    """
    Generalized Unscented Kalman Filter (UKF).
    Supports states on manifolds by providing custom add, subtract, and mean functions.
    """
    def __init__(self, dim_x, dim_z, dim_p=None, alpha=1e-3, beta=2.0, kappa=0.0, 
                 subtract_x=None, add_x=None, mean_x=None):
        """
        Initialize the UKF.
        dim_x: Dimension of the state vector x
        dim_z: Dimension of the measurement vector z
        dim_p: Dimension of the covariance matrix P (and tangent space). Defaults to dim_x.
        alpha, beta, kappa: UKF tuning parameters
        subtract_x: Function (x1, x2) -> dx (difference in tangent space, vector of size dim_p)
        add_x: Function (x, dx) -> x_new (add tangent vector dx of size dim_p to state x)
        mean_x: Function (sigmas, weights) -> x_mean (average sigma points of size dim_x)
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_p = dim_p if dim_p is not None else dim_x
        
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # Scaling parameter based on tangent space dimension
        self.lambda_ = alpha**2 * (self.dim_p + kappa) - self.dim_p
        self.gamma = np.sqrt(self.dim_p + self.lambda_)
        self.num_sigmas = 2 * self.dim_p + 1
        
        # Weights for mean and covariance
        self.Wm = np.zeros(self.num_sigmas)
        self.Wc = np.zeros(self.num_sigmas)
        
        self.Wm[0] = self.lambda_ / (self.dim_p + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.dim_p + self.lambda_) + (1 - alpha**2 + beta)
        
        w = 1.0 / (2 * (self.dim_p + self.lambda_))
        for i in range(1, self.num_sigmas):
            self.Wm[i] = w
            self.Wc[i] = w
            
        # Default vector operations if none provided (assumes dim_p == dim_x)
        self.subtract_x = subtract_x if subtract_x is not None else lambda x1, x2: x1 - x2
        self.add_x = add_x if add_x is not None else lambda x, dx: x + dx
        self.mean_x = mean_x if mean_x is not None else lambda sigmas, weights: np.dot(weights, sigmas)

        self.x = np.zeros(dim_x)
        self.P = np.eye(self.dim_p)
        self.Q = np.eye(self.dim_p)
        self.R = np.eye(dim_z)

    def predict(self, dt, fx, Q=None, **kwargs):
        """
        Predict step.
        dt: Time step
        fx: State transition function f(x, dt, **kwargs) -> x_new
        Q: Optional process noise covariance
        """
        if Q is None: Q = self.Q
        
        # Generate sigma points from current state
        sigmas = self.generate_sigma_points(self.x, self.P)
        
        # Propagate sigma points
        sigmas_f = []
        for i in range(self.num_sigmas):
            sigmas_f.append(fx(sigmas[i], dt, **kwargs))
        sigmas_f = np.array(sigmas_f)
            
        # Calculate predicted mean
        self.x = self.mean_x(sigmas_f, self.Wm)
        
        # Calculate predicted covariance
        self.P = np.zeros((self.dim_p, self.dim_p))
        for i in range(self.num_sigmas):
            dx = self.subtract_x(sigmas_f[i], self.x)
            self.P += self.Wc[i] * np.outer(dx, dx)
        self.P += Q * dt

    def update(self, z, hx, R=None, **kwargs):
        """
        Update step.
        z: Measurement vector
        hx: Measurement function h(x, **kwargs) -> z_pred
        R: Optional measurement noise covariance
        """
        if R is None: R = self.R
        
        # Regenerate sigma points from predicted distribution
        sigmas_f = self.generate_sigma_points(self.x, self.P)
        
        # Transform sigma points into measurement space
        sigmas_h = []
        for i in range(self.num_sigmas):
            sigmas_h.append(hx(sigmas_f[i], **kwargs))
        sigmas_h = np.array(sigmas_h)
            
        # Mean measurement
        zp = np.dot(self.Wm, sigmas_h)
        
        # Innovation covariance S and Cross-covariance Pxz
        S = np.zeros((self.dim_z, self.dim_z))
        Pxz = np.zeros((self.dim_p, self.dim_z))
        
        for i in range(self.num_sigmas):
            dz = sigmas_h[i] - zp
            dx = self.subtract_x(sigmas_f[i], self.x)
            
            S += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)
            
        S += R
        
        # Kalman Gain
        K = np.dot(Pxz, np.linalg.inv(S))
        
        # Correct state and covariance
        innov = z - zp
        self.x = self.add_x(self.x, np.dot(K, innov))
        self.P = self.P - np.dot(K, np.dot(S, K.T))

    def generate_sigma_points(self, x, P):
        """Generates sigma points around x using covariance P in tangent space."""
        sigmas = [x]
        
        # Ensure symmetry and add small epsilon for stability
        P_sym = (P + P.T) / 2 + np.eye(self.dim_p) * 1e-12
        
        try:
            L = cholesky((self.dim_p + self.lambda_) * P_sym, lower=True)
            for i in range(self.dim_p):
                sigmas.append(self.add_x(x, L[:, i]))
                sigmas.append(self.add_x(x, -L[:, i]))
        except np.linalg.LinAlgError:
            # Fallback for non-PSD matrices
            U = sqrtm((self.dim_p + self.lambda_) * P_sym).real
            for i in range(self.dim_p):
                sigmas.append(self.add_x(x, U[i]))
                sigmas.append(self.add_x(x, -U[i]))
            
        return np.array(sigmas)

class UKF_Attitude(UKF):
    """
    Specialized UKF for Attitude Estimation.
    State: [q0, q1, q2, q3, bias_x, bias_y, bias_z] (7 dim)
    Covariance/Error: 6 dim (tangent space)
    """
    def __init__(self, q_init=None, bias_init=None, **kwargs):
        from gnc_toolkit.utils.quat_utils import quat_mult, quat_conj, axis_angle_to_quat, quat_normalize
        
        self._quat_mult = quat_mult
        self._quat_conj = quat_conj
        self._axis_angle_to_quat = axis_angle_to_quat
        self._quat_normalize = quat_normalize

        def subtract_x(x1, x2):
            # q1 = q2 * dq => dq = q2_conj * q1
            dq = self._quat_mult(self._quat_conj(x2[:4]), x1[:4])
            if dq[3] < 0: dq *= -1 
            dtheta = 2 * dq[:3] 
            dbias = x1[4:] - x2[4:]
            return np.concatenate([dtheta, dbias])

        def add_x(x, dx):
            dq = self._axis_angle_to_quat(dx[:3])
            # q_new = q_old * dq
            q_new = self._quat_normalize(self._quat_mult(x[:4], dq))
            bias_new = x[4:] + dx[3:]
            return np.concatenate([q_new, bias_new])

        def mean_x(sigmas, weights):
            # Simple renormalized weighted mean for quaternions
            q_ref = sigmas[0, :4]
            q_avg = np.zeros(4)
            for i in range(len(weights)):
                q = sigmas[i, :4]
                if np.dot(q, q_ref) < 0: q = -q # Consistent hemisphere
                q_avg += weights[i] * q
            
            q_avg = self._quat_normalize(q_avg)
            bias_avg = np.dot(weights, sigmas[:, 4:])
            return np.concatenate([q_avg, bias_avg])

        # Default to a small alpha for better local linearity on manifolds
        if 'alpha' not in kwargs:
            kwargs['alpha'] = 1e-2
            
        super().__init__(dim_x=7, dim_z=3, dim_p=6, 
                         subtract_x=subtract_x, add_x=add_x, mean_x=mean_x, **kwargs)
        
        if q_init is None: q_init = np.array([0, 0, 0, 1.0])
        if bias_init is None: bias_init = np.zeros(3)
        self.x = np.concatenate([q_init, bias_init])
