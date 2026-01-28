import numpy as np

class EKF:
    """
    Extended Kalman Filter (EKF).
    Suitable for non-linear estimation and navigation.
    """
    def __init__(self, dim_x, dim_z):
        """
        Initialize the EKF.
        dim_x: Dimension of the state vector
        dim_z: Dimension of the measurement vector
        """
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)

    def predict(self, fx, F_jac, dt, u=None, Q=None, **kwargs):
        """
        Predict step.
        fx: State transition function f(x, dt, u, **kwargs) -> x_new
        F_jac: Function that returns the Jacobian of f at (x, dt, u) -> F matrix
        dt: Time step
        u: Optional control input
        Q: Optional process noise covariance
        """
        if Q is None: Q = self.Q

        # State Prediction: x = f(x, dt, u)
        self.x = fx(self.x, dt, u, **kwargs)

        # Covariance Prediction: P = FPF' + Q
        F = F_jac(self.x, dt, u, **kwargs)
        self.P = np.dot(np.dot(F, self.P), F.T) + Q

    def update(self, z, hx, H_jac, R=None, **kwargs):
        """
        Update step.
        z: Measurement vector
        hx: Measurement function h(x, **kwargs) -> z_pred
        H_jac: Function that returns the Jacobian of h at x -> H matrix
        R: Optional measurement noise covariance
        """
        if R is None: R = self.R

        # Innovation: y = z - h(x)
        y = z - hx(self.x, **kwargs)

        # Jacobian H
        H = H_jac(self.x, **kwargs)
            
        # Innovation Covariance: S = HPH' + R
        S = np.dot(np.dot(H, self.P), H.T) + R
        
        # Kalman Gain: K = PH'S^-1
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        
        # State Correction: x = x + Ky
        self.x = self.x + np.dot(K, y)
        
        # Covariance Correction: P = (I - KH)P(I - KH)' + KRK' (Joseph Form)
        I = np.eye(self.dim_x)
        I_KH = I - np.dot(K, H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(K, R), K.T)
