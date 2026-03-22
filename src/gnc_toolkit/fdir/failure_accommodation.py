"""
Actuator accommodation and control allocation weight adjustment.
"""

import numpy as np
from typing import List, Optional

class ActuatorAccommodation:
    """
    Accommodates actuator failures by updating control allocation weights.
    
    Standard control allocation problem:
        tau = B * u
    where:
        tau: Desired control effort (e.g., torque)
        u: Actuator commands
        B: Control allocation / geometry matrix (k x m)
        
    We want to find u such that tau = B * u while minimizing some cost, 
    often u^T * W * u, where W is a weight matrix.
    
    Minimum norm solution with weights:
        u = W^{-1} * B^T * (B * W^{-1} * B^T)^{-1} * tau
        
    We can also use a "health" vector h (m x 1) where h_i in [0, 1].
    If h_i = 0, actuator i is failed and we set its weight to infinity 
    (or remove it from the allocation).
    """
    def __init__(self, B: np.ndarray, initial_weights: Optional[np.ndarray] = None):
        """
        Initialize the actuator accommodation list.
        
        Args:
            B: Control allocation matrix (k x m)
            initial_weights: Weight matrix W (m x m) or diagonal elements (m x 1).
                             Defaults to Identity (equal weighting).
        """
        self.B = B
        self.k, self.m = B.shape
        
        if initial_weights is None:
            self.W_diag = np.ones(self.m)
        else:
            if initial_weights.ndim == 2:
                self.W_diag = np.diag(initial_weights)
            else:
                self.W_diag = initial_weights
                
        self.health = np.ones(self.m)  # 1 = Healthy, 0 = Failed
        
    def set_health(self, index: int, status: float):
        """
        Set the health status of an actuator.
        
        Args:
            index: Actuator index (0 to m-1)
            status: Health status (1.0 = healthy, 0.0 = completely failed)
        """
        if index < 0 or index >= self.m:
            raise IndexError(f"Actuator index {index} out of bounds")
        self.health[index] = status
        
    def update_allocation_matrix(self, W_diag: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the pseudo-inverse allocation matrix based on current weights and health.
        
        Returns:
            B_pinv: Pseudo-inverse matrix (m x k)
        """
        weights = self.W_diag.copy() if W_diag is None else W_diag.copy()
        
        # Apply health status
        effective_weights = weights.copy()
        for i in range(self.m):
            if self.health[i] <= 1e-6:
                effective_weights[i] = 1e12  # High cost to use this actuator
            else:
                effective_weights[i] = weights[i] / self.health[i]
                
        W_inv_diag = 1.0 / effective_weights
        W_inv = np.diag(W_inv_diag)
        
        # Calculate B_pinv = W_inv * B^T * (B * W_inv * B^T)^{-1}
        try:
            temp = self.B @ W_inv @ self.B.T
            B_pinv = W_inv @ self.B.T @ np.linalg.inv(temp)
        except np.linalg.LinAlgError:
            # Fallback to standard pinv if singular
            B_pinv = np.linalg.pinv(self.B)
            
        return B_pinv
        
    def allocate(self, tau: np.ndarray) -> np.ndarray:
        """
        Allocate control effort to actuators.
        
        Args:
            tau: Desired control effort (k x 1)
            
        Returns:
            u: Actuator commands (m x 1)
        """
        tau = tau.reshape(-1, 1)
        B_pinv = self.update_allocation_matrix()
        return B_pinv @ tau
