"""
Model Reference Adaptive Control (MRAC) for state-space systems.
"""

import numpy as np
from scipy.linalg import solve_continuous_lyapunov

class ModelReferenceAdaptiveControl:
    """
    Model Reference Adaptive Controller (MRAC) for state-space systems.
    
    Plant with parametric uncertainty:
    dx/dt = A * x + B * (u + theta^T * phi(x))
    
    Reference Model:
    dx_m/dt = A_m * x_m + B_m * r
    
    Adaptive estimate update law:
    d_theta_hat/dt = Gamma * phi(x) * e^T * P * B
    
    where e = x - x_m, and P solves A_m^T * P + P * A_m = -Q.
    """
    def __init__(self, A_m, B_m, B, Gamma, Q_lyap, phi_func):
        """
        Initialize the MRAC.
        
        Args:
            A_m (np.ndarray): Reference model system matrix [n x n].
            B_m (np.ndarray): Reference model input matrix [n x m].
            B (np.ndarray): Plant input matrix [n x m].
            Gamma (np.ndarray): Adaptation rate matrix [k x k] where k is num parameters.
            Q_lyap (np.ndarray): positive-definite Q for Lyapunov equation [n x n].
            phi_func (callable): Regressor function phi(x) -> vector of size [k].
        """
        self.A_m = np.array(A_m)
        self.B_m = np.array(B_m)
        self.B = np.array(B)
        self.Gamma = np.array(Gamma)
        self.phi = phi_func
        
        # Solve Lyapunov Equation: A_m.T * P + P * A_m = -Q
        self.P = solve_continuous_lyapunov(self.A_m.T, -np.array(Q_lyap))
        
        # Initial estimate of theta (unknown parameters)
        self.nx = self.A_m.shape[0]
        self.nu = self.B_m.shape[1]
        
        # Verify dimension of phi
        test_x = np.zeros(self.nx)
        test_phi = self.phi(test_x)
        self.k_params = len(test_phi)
        
        # Parameter estimate matrix [k_params, nu]
        self.theta_hat = np.zeros((self.k_params, self.nu))

    def compute_control(self, x, x_m, r, kx=None, kr=None):
        """
        Compute control input u.
        
        Args:
            x (np.ndarray): Current plant state [n].
            x_m (np.ndarray): Current reference model state [n].
            r (np.ndarray): Reference input [m].
            kx (np.ndarray, optional): Optimal gain for A + B*kx = A_m. Defaults to using values that assume A=A_m.
            kr (np.ndarray, optional): Optimal gain for B*kr = B_m. Defaults to ones.
            
        Returns:
            np.ndarray: Control effort u [m].
        """
        x = np.array(x)
        x_m = np.array(x_m)
        r = np.array(r)
        
        # Defaults if not provided (ideal matching gains)
        if kx is None:
            kx = np.zeros((self.nu, self.nx))  # Assume A = A_m
        if kr is None:
            kr = np.eye(self.nu)                # Assume B = B_m
            
        phi_x = self.phi(x).reshape(-1, 1)  # [k x 1]
        adaptive_term = (self.theta_hat.T @ phi_x).flatten()
        u = kx @ x + kr @ r - adaptive_term
        
        # Parameter derivative for Euler update: d_theta = Gamma * phi(x) * e.T * P * B
        e = x - x_m
        error_term = e.reshape(1, -1) @ self.P @ self.B  # [1 x m]
        d_theta = self.Gamma @ phi_x @ error_term         # [k x m]
        self.d_theta_hat = d_theta
        
        return u

    def update_theta(self, dt):
        """
        Update the parameter estimate using Euler integration.
        Should be called after compute_control and plant simulation step.
        """
        if hasattr(self, 'd_theta_hat'):
             self.theta_hat += self.d_theta_hat * dt
