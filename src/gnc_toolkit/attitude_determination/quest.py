import numpy as np
from gnc_toolkit.utils.quat_utils import quat_normalize, quat_conj

def quest(body_vectors, ref_vectors, weights=None):
    """
    Compute the optimal attitude quaternion (Inertial -> Body) using the QUEST algorithm.
    
    Args:
        body_vectors (list or np.ndarray): List of N vectors measured in body frame. Shape (N, 3).
        ref_vectors (list or np.ndarray): List of N vectors known in reference (inertial) frame. Shape (N, 3).
        weights (list or np.ndarray): List of N scalar weights. If None, weights are equal. Shape (N,).
                                      
    Returns:
        np.ndarray: Normalized quaternion [x, y, z, w] representing rotation R_BI.
    """
    b_vecs = np.asarray(body_vectors)
    r_vecs = np.asarray(ref_vectors)
    
    n = b_vecs.shape[0]
    if r_vecs.shape[0] != n:
        raise ValueError("Number of body vectors must match number of reference vectors.")
        
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.asarray(weights)
        
    # Normalize input vectors
    for i in range(n):
        b_vecs[i] = b_vecs[i] / np.linalg.norm(b_vecs[i])
        r_vecs[i] = r_vecs[i] / np.linalg.norm(r_vecs[i])

    # Compute B matrix
    B = np.zeros((3, 3))
    for i in range(n):
        # outer product: b * r^T
        B += weights[i] * np.outer(b_vecs[i], r_vecs[i])
        
    S = B + B.T
    
    # Z vector = [B23 - B32, B31 - B13, B12 - B21]^T
    Z = np.array([
        B[1, 2] - B[2, 1],
        B[2, 0] - B[0, 2],
        B[0, 1] - B[1, 0]
    ])
    
    sigma = np.trace(B)
    
    # Construct K matrix (4x4)
    # K = [ S - sigma*I    Z ]
    #     [    Z^T       sigma ]
    
    K = np.zeros((4, 4))
    K[0:3, 0:3] = S - sigma * np.eye(3)
    K[0:3, 3] = Z 
    K[3, 0:3] = Z
    K[3, 3] = sigma
    
    # Solve for eigenvalues/eigenvectors
    vals, vecs = np.linalg.eigh(K)
    
    # eigh returns eigenvalues in ascending order
    # The last one is the largest
    max_idx = np.argmax(vals)
    q_opt = vecs[:, max_idx]
    
    return quat_normalize(quat_conj(q_opt))
