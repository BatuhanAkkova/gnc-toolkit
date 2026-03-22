"""
Residual generation for fault detection using observers.
"""

import numpy as np
from typing import Callable, Optional

class ObserverResidualGenerator:
    """
    Generates residuals using a Luenberger observer or similar linear observer.
    
    System model:
        x_{k+1} = A x_k + B u_k
        y_k     = C x_k + D u_k
        
    Observer:
        x_hat_{k+1} = A x_hat_k + B u_k + L (y_k - y_hat_k)
        y_hat_k     = C x_hat_k + D u_k
        
    Residual:
        r_k = y_k - y_hat_k
    """
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: Optional[np.ndarray], L: np.ndarray, x0: Optional[np.ndarray] = None):
        """
        Initialize the observer.
        
        Args:
            A: State transition matrix (n x n)
            B: Input matrix (n x m)
            C: Output matrix (p x n)
            D: Direct feedthrough matrix (p x m), defaults to zeros if None
            L: Observer gain matrix (n x p)
            x0: Initial state estimate (n x 1), defaults to zeros
        """
        self.A = A
        self.B = B
        self.C = C
        self.D = D if D is not None else np.zeros((C.shape[0], B.shape[1]))
        self.L = L
        
        self.n = A.shape[0]
        self.m = B.shape[1]
        self.p = C.shape[0]
        
        self.x_hat = x0 if x0 is not None else np.zeros((self.n, 1))
        
    def step(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Perform one step of the observer and return the residual.
        
        Args:
            u: Input vector (m x 1)
            y: Measurement vector (p x 1)
            
        Returns:
            r: Residual vector (p x 1)
        """
        # Ensure column vectors
        u = u.reshape(-1, 1)
        y = y.reshape(-1, 1)
        
        # Calculate estimate output
        y_hat = self.C @ self.x_hat + self.D @ u
        
        # Calculate residual
        r = y - y_hat
        
        # Update state estimate
        self.x_hat = self.A @ self.x_hat + self.B @ u + self.L @ r
        
        return r

class AnalyticalRedundancy:
    """
    Detects faults by comparing two signals that should be algebraically or dynamically related.
    
    Example: Comparing integrated gyro rates with star tracker quaternions,
    or comparing two redundant sensor outputs.
    """
    @staticmethod
    def check_threshold(r: np.ndarray, threshold: float) -> bool:
        """
        Checks if the residual exceeds a threshold.
        
        Args:
            r: Residual vector
            threshold: Scalar threshold
            
        Returns:
            True if fault detected, False otherwise
        """
        return np.linalg.norm(r) > threshold

    @staticmethod
    def gyro_vs_quaternion_residual(q_dot_measured: np.ndarray, q_dot_calculated: np.ndarray) -> np.ndarray:
        """
        Calculates residual between measured quaternion rate (from ST) and calculated from gyro.
        
        q_dot_calculated = 0.5 * q \otimes [0, \omega]^T
        
        Args:
            q_dot_measured: Measured quaternion rate
            q_dot_calculated: Calculated quaternion rate
            
        Returns:
            Residual vector
        """
        return q_dot_measured - q_dot_calculated
