import numpy as np
from gnc_toolkit.utils.quat_utils import quat_normalize

def davenport_q(body_vectors, ref_vectors, weights=None):
    """
    Compute the optimal attitude quaternion (Inertial -> Body) using Davenport's q-method.
    
    This method solves the Wahba problem by finding the eigenvector corresponding 
    to the largest eigenvalue of the Davenport K-matrix.

    Args:
        body_vectors (list or np.ndarray): List of N vectors measured in body frame. Shape (N, 3).
        ref_vectors (list or np.ndarray): List of N vectors known in reference (inertial) frame. Shape (N, 3).
        weights (list or np.ndarray, optional): List of N scalar weights. If None, weights are equal.
                                       
    Returns:
        np.ndarray: Normalized quaternion [x, y, z, w] representing rotation R_BI.
    """
    b_vecs = np.asarray(body_vectors)
    r_vecs = np.asarray(ref_vectors)
    
    if b_vecs.shape != r_vecs.shape:
        raise ValueError("Body and reference vector arrays must have the same shape.")
        
    n = b_vecs.shape[0]
    if weights is None:
        weights = np.ones(n) / n
    else:
        weights = np.asarray(weights)
        if len(weights) != n:
            raise ValueError("Number of weights must match number of vectors.")

    # Normalize input vectors (Wahba problem assumes unit vectors)
    b_vecs_norm = b_vecs / np.linalg.norm(b_vecs, axis=1)[:, np.newaxis]
    r_vecs_norm = r_vecs / np.linalg.norm(r_vecs, axis=1)[:, np.newaxis]

    # Compute Attitude Profile Matrix B = sum(w_i * b_i * r_i^T)
    B = np.zeros((3, 3))
    for i in range(n):
        B += weights[i] * np.outer(b_vecs_norm[i], r_vecs_norm[i])

    S = B + B.T
    sigma = np.trace(B)
    Z = np.array([
        B[1, 2] - B[2, 1],
        B[2, 0] - B[0, 2],
        B[0, 1] - B[1, 0]
    ])

    # Construct Davenport K-matrix
    K = np.zeros((4, 4))
    K[0:3, 0:3] = S - sigma * np.eye(3)
    K[0:3, 3] = Z
    K[3, 0:3] = Z
    K[3, 3] = sigma

    # Solve for eigenvalues/eigenvectors
    # K is symmetric, so we use eigh
    vals, vecs = np.linalg.eigh(K)
    
    # Largest eigenvalue corresponds to the optimal quaternion
    q_opt = vecs[:, np.argmax(vals)]
    
    return quat_normalize(q_opt)
