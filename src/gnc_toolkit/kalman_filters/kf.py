import numpy as np

class KF:
    """
    Standard Linear Kalman Filter (KF).
    Suitable for linear estimation and navigation (e.g., constant velocity models).
    """
    def __init__(self, dim_x, dim_z):
        """
        Initialize the KF.
        dim_x: Dimension of the state vector
        dim_z: Dimension of the measurement vector
        """
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.F = np.eye(dim_x)
        self.H = np.zeros((dim_z, dim_x))
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)
        self.B = None

    def predict(self, u=None, F=None, Q=None, B=None):
        """
        Predict step.
        u: Control input vector
        F: State transition matrix
        Q: Process noise covariance
        B: Control input matrix
        """
        if F is None: F = self.F
        if Q is None: Q = self.Q
        if B is None: B = self.B

        # x = Fx + Bu
        if B is not None and u is not None:
            self.x = np.dot(F, self.x) + np.dot(B, u)
        else:
            self.x = np.dot(F, self.x)

        # P = FPF' + Q
        self.P = np.dot(np.dot(F, self.P), F.T) + Q

    def update(self, z, H=None, R=None):
        """
        Update step.
        z: Measurement vector
        H: Measurement matrix
        R: Measurement noise covariance
        """
        if H is None: H = self.H
        if R is None: R = self.R

        # y = z - Hx
        y = z - np.dot(H, self.x)

        # S = HPH' + R
        S = np.dot(np.dot(H, self.P), H.T) + R

        # K = PH'S^-1
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))

        # x = x + Ky
        self.x = self.x + np.dot(K, y)

        # P = (I - KH)P(I - KH)' + KRK' (Joseph Form for stability)
        I = np.eye(self.dim_x)
        I_KH = I - np.dot(K, H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(K, R), K.T)
