"""
Parity Space methods for Fault Detection and Isolation (FDI).
"""

import numpy as np
from typing import Tuple, List

class ParitySpaceDetector:
    """
    Implements Parity Space methods for Fault Detection and Isolation (FDI) 
    in redundant sensor systems.
    
    Measurement model:
        y = M * x + v + f
    where:
        y: Measurement vector (p x 1)
        M: Geometry/Mixing matrix (p x n, p > n)
        x: True state (n x 1)
        v: Measurement noise
        f: Fault vector
        
    Parity space matrix P satisfies:
        P * M = 0
        P * P^T = I
    Parity vector:
        p_vec = P * y
    """
    def __init__(self, M: np.ndarray):
        """
        Initialize the parity space detector.
        
        Args:
            M: Measurement matrix (p x n) with p > n
        """
        self.M = M
        self.p_dim, self.n_dim = M.shape
        
        if self.p_dim <= self.n_dim:
            raise ValueError("Parity space requires redundant measurements (p > n)")
            
        # Compute P matrix using SVD
        U, S, Vh = np.linalg.svd(M)
        self.P = U[:, self.n_dim:].T  # Shape: (p-n) x p
        
    def get_parity_vector(self, y: np.ndarray) -> np.ndarray:
        """
        Calculate the parity vector for a given measurement.
        
        Args:
            y: Measurement vector (p x 1)
            
        Returns:
            p_vec: Parity vector (p-n x 1)
        """
        y = y.reshape(-1, 1)
        return self.P @ y
        
    def detect_fault(self, y: np.ndarray, threshold: float) -> bool:
        """
        Detect if a fault is present.
        
        Args:
            y: Measurement vector
            threshold: Magnitude threshold for the parity vector norm
            
        Returns:
            True if fault detected, False otherwise
        """
        p_vec = self.get_parity_vector(y)
        return np.linalg.norm(p_vec) > threshold
        
    def isolate_fault(self, y: np.ndarray) -> int:
        """
        Isolate which sensor is faulty by finding the column of P 
        that best aligns with the parity vector.
        
        Args:
            y: Measurement vector
            
        Returns:
            Isolated sensor index (0 to p-1)
        """
        p_vec = self.get_parity_vector(y).flatten()
        
        if np.linalg.norm(p_vec) == 0:
            return -1  # No fault
            
        # Normalize parity vector
        p_norm = p_vec / np.linalg.norm(p_vec)
        
        # Calculate alignment with columns of P
        alignments = []
        for i in range(self.p_dim):
            P_col = self.P[:, i]
            if np.linalg.norm(P_col) > 0:
                P_col_norm = P_col / np.linalg.norm(P_col)
                alignments.append(np.abs(np.dot(p_norm, P_col_norm)))
            else:
                alignments.append(0.0)
                
        return int(np.argmax(alignments))
