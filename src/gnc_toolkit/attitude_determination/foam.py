"""
Fast Optimal Attitude Matrix (FOAM) algorithm for attitude determination.
"""

import numpy as np


from typing import Optional, Union

def foam(
    body_vectors: np.ndarray,
    ref_vectors: np.ndarray,
    weights: Optional[np.ndarray] = None,
    tol: float = 1e-12,
    max_iter: int = 20,
) -> np.ndarray:
    r"""
    Solve for the optimal attitude matrix using FOAM.

    Directly computes the DCM:
    $\mathbf{R}_{BI} = \frac{(\lambda^2 + \|\mathbf{B}\|_F^2)\mathbf{B} + 2\lambda \text{adj}(\mathbf{B})^T - 2\mathbf{B} \mathbf{B}^T \mathbf{B}}{\lambda(\lambda^2 - \|\mathbf{B}\|_F^2) - 2 \det(\mathbf{B})}$

    Parameters
    ----------
    body_vectors : np.ndarray
        Body measurements (N, 3).
    ref_vectors : np.ndarray
        Inertial references (N, 3).
    weights : np.ndarray | None, optional
        Weights (N,).
    tol : float, optional
        Tolerance. Default 1e-12.
    max_iter : int, optional
        Max iterations. Default 20.

    Returns
    -------
    np.ndarray
        Optimal $3 \times 3$ DCM $\mathbf{R}_{BI}$.
    """
    b_vecs = np.asarray(body_vectors)
    r_vecs = np.asarray(ref_vectors)

    if b_vecs.shape != r_vecs.shape:
        raise ValueError("Body and reference vector arrays must have the same shape.")

    n_vecs = b_vecs.shape[0]
    w = np.asarray(weights) if weights is not None else np.ones(n_vecs) / n_vecs
    if len(w) != n_vecs:
        raise ValueError("Number of weights must match number of vectors.")

    # Normalize vectors and compute profile matrix B
    b_norm = b_vecs / np.linalg.norm(b_vecs, axis=1)[:, np.newaxis]
    r_norm = r_vecs / np.linalg.norm(r_vecs, axis=1)[:, np.newaxis]

    b_matrix = np.zeros((3, 3))
    for i in range(n_vecs):
        b_matrix += w[i] * np.outer(b_norm[i], r_norm[i])

    det_b = float(np.linalg.det(b_matrix))
    # Correct adjugate calculation
    adj_b = np.zeros((3, 3))
    adj_b[0, 0] = b_matrix[1, 1]*b_matrix[2, 2] - b_matrix[1, 2]*b_matrix[2, 1]
    adj_b[0, 1] = b_matrix[0, 2]*b_matrix[2, 1] - b_matrix[0, 1]*b_matrix[2, 2]
    adj_b[0, 2] = b_matrix[0, 1]*b_matrix[1, 2] - b_matrix[0, 2]*b_matrix[1, 1]
    adj_b[1, 0] = b_matrix[1, 2]*b_matrix[2, 0] - b_matrix[1, 0]*b_matrix[2, 2]
    adj_b[1, 1] = b_matrix[0, 0]*b_matrix[2, 2] - b_matrix[0, 2]*b_matrix[2, 0]
    adj_b[1, 2] = b_matrix[0, 2]*b_matrix[1, 0] - b_matrix[0, 0]*b_matrix[1, 2]
    adj_b[2, 0] = b_matrix[1, 0]*b_matrix[2, 1] - b_matrix[1, 1]*b_matrix[2, 0]
    adj_b[2, 1] = b_matrix[0, 1]*b_matrix[2, 0] - b_matrix[0, 0]*b_matrix[2, 1]
    adj_b[2, 2] = b_matrix[0, 0]*b_matrix[1, 1] - b_matrix[0, 1]*b_matrix[1, 0]

    b_frob_sq = float(np.trace(b_matrix @ b_matrix.T))
    adj_b_frob_sq = float(np.trace(adj_b @ adj_b.T))

    # Solve for lambda_max using Newton-Raphson
    lam = float(np.sum(w))
    for _ in range(max_iter):
        f_val = (lam**2 - b_frob_sq)**2 - 8*lam*det_b - 4*adj_b_frob_sq
        fp_val = 4*lam*(lam**2 - b_frob_sq) - 8*det_b
        
        delta = f_val / fp_val
        lam -= delta
        if abs(delta) < tol:
            break

    # Construct the final DCM directly using Markley's robust algorithm
    # R_opt = [ (l^2 + ||B||^2)B + 2*l*adj(B)^T - 2*B*B^T*B ] / [ l(l^2 - ||B||^2) - 2*det(B) ]
    num = (lam**2 + b_frob_sq) * b_matrix + 2 * lam * adj_b.T - 2 * (b_matrix @ b_matrix.T @ b_matrix)
    den = lam * (lam**2 - b_frob_sq) - 2 * det_b

    return num / den
