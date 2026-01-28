import numpy as np
from gnc_toolkit.utils.quat_utils import quat_mult, quat_normalize, quat_conj, quat_rot, skew_symmetric

class MEKF:
    """
    Multiplicative Extended Kalman Filter (MEKF) for Attitude Estimation.
    Reference: Crassidis and Junkins, "Optimal Estimation of Dynamic Systems".
    State: [q_x, q_y, q_z, q_w, beta_x, beta_y, beta_z] (7x1)
    Error State: [delta_theta_x, delta_theta_y, delta_theta_z, delta_beta_x, delta_beta_y, delta_beta_z] (6x1)
    """
    def __init__(self, q_init=None, beta_init=None):
        """
        Initialize the MEKF.
        q_init: Initial quaternion [q_x, q_y, q_z, q_w]
        beta_init: Initial gyro bias [bx, by, bz]
        """
        if q_init is None:
            self.q = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            self.q = quat_normalize(q_init)
            
        if beta_init is None:
            self.beta = np.zeros(3)
        else:
            self.beta = np.asarray(beta_init)

        # Covariance (6x6 for error state)
        self.P = np.eye(6) * 0.1
        
        # Noise Covariances (standard defaults)
        self.Q = np.eye(6) * 0.001 
        self.R = np.eye(3) * 0.01  

        # Internal state vector (for consistency with other filters)
        self.x = np.concatenate([self.q, self.beta])

    def predict(self, omega_meas, dt, Q=None):
        """
        MEKF Prediction Step.
        omega_meas: Measured angular velocity (3x1)
        dt: Time step
        Q: Optional process noise covariance
        """
        if Q is None: Q = self.Q

        # Reference State Integration (Body frame integration: q_new = q_old * dq)
        omega = omega_meas - self.beta
        omega_norm = np.linalg.norm(omega)
        
        if omega_norm > 1e-10:
            axis = omega / omega_norm
            angle = omega_norm * dt
            dq = np.concatenate([axis * np.sin(angle/2), [np.cos(angle/2)]])
            self.q = quat_mult(self.q, dq) # Right multiplication
        
        self.q = quat_normalize(self.q)
        
        # Covariance Prediction (Body frame error state)
        wx = skew_symmetric(omega)
        F = np.zeros((6, 6))
        F[0:3, 0:3] = -wx 
        F[0:3, 3:6] = -np.eye(3)
        
        Phi = np.eye(6) + F * dt
        self.P = np.dot(np.dot(Phi, self.P), Phi.T) + Q * dt
        
        self.x = np.concatenate([self.q, self.beta])

    def update(self, z, z_ref, R=None):
        """
        MEKF Update Step.
        z: Measured vector in body frame (normalized)
        z_ref: Reference vector in inertial frame (normalized)
        R: Optional measurement noise covariance
        """
        if R is None: R = self.R
        
        # Predicted measurement in body frame: z_pred = R(q)^T * z_ref
        q_inv = quat_conj(self.q)
        z_pred = quat_rot(q_inv, z_ref)
        
        # Innovation: y = z_meas - z_pred
        y = z - z_pred
        
        # Sensitivity matrix H = [ [z_pred]x  0 ]
        H = np.zeros((3, 6))
        H[:, 0:3] = skew_symmetric(z_pred)
        
        # Innovation Covariance S and Kalman Gain K
        S = np.dot(np.dot(H, self.P), H.T) + R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        
        # Error state correction
        dx = np.dot(K, y)
        dtheta = dx[0:3]
        dbeta = dx[3:6]
        
        # Apply corrections (Multiplicative for quat, Additive for bias)
        dq_corr = np.concatenate([0.5 * dtheta, [1.0]])
        self.q = quat_normalize(quat_mult(self.q, dq_corr)) # Right correction
        self.beta += dbeta
        
        # Covariance Update (Joseph Form)
        I = np.eye(6)
        I_KH = I - np.dot(K, H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(K, R), K.T)
        
        self.x = np.concatenate([self.q, self.beta])
