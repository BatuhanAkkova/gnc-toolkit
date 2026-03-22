"""
Linear Quadratic Gaussian (LQG) Controller.
"""

import numpy as np
from .lqr import LQR
from .lqe import LQE

class LQG:
    """
    Linear Quadratic Gaussian (LQG) Controller.
    
    Combines LQR for optimal feedback and LQE (Kalman Filter) for optimal estimation.
    The separation principle allows these to be designed independently.
    
    System model:
    x_dot = A*x + B*u + G*w
    y = C*x + v
    
    Control law:
    u = -K * x_hat
    """
    def __init__(self, A, B, C, Q_lqr, R_lqr, Q_lqe, R_lqe, G_lqe=None):
        """
        Initialize the LQG controller.
        
        Args:
            A (np.ndarray): State matrix (nx x nx)
            B (np.ndarray): Input matrix (nx x nu)
            C (np.ndarray): Output matrix (ny x nx)
            Q_lqr (np.ndarray): State cost matrix for LQR
            R_lqr (np.ndarray): Input cost matrix for LQR
            Q_lqe (np.ndarray): Process noise covariance for LQE
            R_lqe (np.ndarray): Measurement noise covariance for LQE
            G_lqe (np.ndarray, optional): Process noise input matrix. Defaults to Identity.
        """
        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)
        
        nx = self.A.shape[0]
        if G_lqe is None:
            G_lqe = np.eye(nx)
            
        # Design LQR
        self.lqr = LQR(A, B, Q_lqr, R_lqr)
        self.K = self.lqr.compute_gain()
        
        # Design LQE
        self.lqe = LQE(A, G_lqe, C, Q_lqe, R_lqe)
        self.L = self.lqe.compute_gain()
        
        # State estimate
        self.x_hat = np.zeros(nx)

    def update_estimation(self, y, u, dt):
        """
        Update the state estimate using the observer dynamics.
        x_hat_dot = A*x_hat + B*u + L*(y - C*x_hat)
        
        Args:
            y (np.ndarray): Measurement
            u (np.ndarray): Last control input
            dt (float): Time step
        """
        # Euler integration for estimation update
        innovation = y - (self.C @ self.x_hat)
        x_hat_dot = (self.A @ self.x_hat) + (self.B @ u) + (self.L @ innovation)
        self.x_hat = self.x_hat + x_hat_dot * dt
        return self.x_hat

    def compute_control(self, y=None, dt=None, u_last=None):
        """
        Compute the control input based on the current state estimate.
        u = -K * x_hat
        
        If y and dt are provided, updates the estimate first.
        
        Args:
            y (np.ndarray, optional): Latest measurement
            dt (float, optional): Time step for estimation update
            u_last (np.ndarray, optional): Last control input for estimation update
            
        Returns:
            np.ndarray: Control input u
        """
        if y is not None and dt is not None:
            # If u_last is omitted, use current estimate-based control
            u_to_est = u_last if u_last is not None else -self.K @ self.x_hat
            self.update_estimation(y, u_to_est, dt)
            
        u = -self.K @ self.x_hat
        return u
