"""
Recursive QUEST (REQUEST) algorithm for recursive attitude estimation.
"""

import numpy as np
from gnc_toolkit.utils.quat_utils import quat_normalize

class RequestFilter:
    """
    Implementation of the REcursive QUEST (REQUEST) algorithm.
    
    REQUEST allows for recursive attitude estimation by updating the K-matrix
    over time, incorporating a fading memory factor.
    """
    def __init__(self, initial_K=None):
        """
        Initialize the REQUEST filter.
        
        Args:
            initial_K (np.ndarray, optional): Initial 4x4 K-matrix. Defaults to zero if None.
        """
        if initial_K is not None:
            self.K = np.asarray(initial_K, dtype=float)
        else:
            self.K = np.zeros((4, 4))
            
    def update(self, body_vectors, ref_vectors, weights=None, rho=1.0):
        """
        Update the K-matrix with new measurements.
        
        Args:
            body_vectors (np.ndarray): N vectors in body frame.
            ref_vectors (np.ndarray): N vectors in reference frame.
            weights (np.ndarray, optional): Weights for each measurement.
            rho (float): Fading memory factor (0 < rho <= 1). 
                         rho=1 means no fading (accumulative).
        
        Returns:
            np.ndarray: Updated 4x4 K-matrix.
        """
        b_vecs = np.asarray(body_vectors)
        r_vecs = np.asarray(ref_vectors)
        n = b_vecs.shape[0]
        
        if weights is None:
            weights = np.ones(n) / n
        else:
            weights = np.asarray(weights)

        # Normalize input vectors
        b_vecs_norm = b_vecs / np.linalg.norm(b_vecs, axis=1)[:, np.newaxis]
        r_vecs_norm = r_vecs / np.linalg.norm(r_vecs, axis=1)[:, np.newaxis]

        # Compute incremental B matrix
        dB = np.zeros((3, 3))
        for i in range(n):
            dB += weights[i] * np.outer(b_vecs_norm[i], r_vecs_norm[i])
            
        dS = dB + dB.T
        d_sigma = np.trace(dB)
        dZ = np.array([
            dB[1, 2] - dB[2, 1],
            dB[2, 0] - dB[0, 2],
            dB[0, 1] - dB[1, 0]
        ])
        
        # Incremental K-matrix
        dK = np.zeros((4, 4))
        dK[0:3, 0:3] = dS - d_sigma * np.eye(3)
        dK[0:3, 3] = dZ
        dK[3, 0:3] = dZ
        dK[3, 3] = d_sigma
        
        # Update K: K_next = rho * K_prev + dK
        self.K = rho * self.K + dK
        return self.K
    
    def get_quaternion(self):
        """
        Extract the optimal quaternion from the current K-matrix.
        
        Returns:
            np.ndarray: Normalized quaternion [x, y, z, w].
        """
        vals, vecs = np.linalg.eigh(self.K)
        q_opt = vecs[:, np.argmax(vals)]
        return quat_normalize(q_opt)

def request(body_vectors, ref_vectors, weights=None, initial_K=None, rho=1.0):
    """
    One-shot REQUEST update helper.
    """
    rf = RequestFilter(initial_K)
    rf.update(body_vectors, ref_vectors, weights, rho)
    return rf.get_quaternion()
