"""
QUEST algorithm for optimal attitude determination (Wahba's problem).
"""

import numpy as np

from gnc_toolkit.utils.quat_utils import quat_normalize


def quest(body_vectors, ref_vectors, weights=None, tol=1e-12, max_iter=20):
    """
    Compute the optimal attitude quaternion (Inertial -> Body) using the QUEST algorithm.

    This implementation uses the iterative approach to find the maximum eigenvalue
    of the K-matrix, which is more efficient than full eigenvalue decomposition.

    Args:
        body_vectors (list or np.ndarray): List of N vectors measured in body frame. Shape (N, 3).
        ref_vectors (list or np.ndarray): List of N vectors known in reference (inertial) frame. Shape (N, 3).
        weights (list or np.ndarray, optional): List of N scalar weights. If None, weights are equal.
        tol (float, optional): Convergence tolerance for the iterative eigenvalue search.
        max_iter (int, optional): Maximum number of iterations for eigenvalue search.

    Returns
    -------
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
    b_vecs_norm = b_vecs / np.linalg.norm(b_vecs, axis=1)[:, np.newaxis]
    r_vecs_norm = r_vecs / np.linalg.norm(r_vecs, axis=1)[:, np.newaxis]

    # Compute B matrix
    B = np.zeros((3, 3))
    for i in range(n):
        B += weights[i] * np.outer(b_vecs_norm[i], r_vecs_norm[i])

    S = B + B.T
    sigma = np.trace(B)
    Z = np.array([B[2, 1] - B[1, 2], B[0, 2] - B[2, 0], B[1, 0] - B[0, 1]])

    # Solve for lambda_max using Newton-Raphson
    # Characteristic polynomial: f(L) = det(K - L*I) = 0
    # Let S' = S - sigma*I
    # f(L) = L^4 - (a+b)L^2 - cL + (ab-d)

    B_sq_norm = np.trace(B @ B.T)
    adj_B = np.array(
        [
            [
                B[1, 1] * B[2, 2] - B[1, 2] * B[2, 1],
                B[0, 2] * B[2, 1] - B[0, 1] * B[2, 2],
                B[0, 1] * B[1, 2] - B[0, 2] * B[1, 1],
            ],
            [
                B[1, 2] * B[2, 0] - B[1, 0] * B[2, 2],
                B[0, 0] * B[2, 2] - B[0, 2] * B[2, 0],
                B[0, 2] * B[1, 0] - B[0, 0] * B[1, 2],
            ],
            [
                B[1, 0] * B[2, 1] - B[1, 1] * B[2, 0],
                B[0, 1] * B[2, 0] - B[0, 0] * B[2, 1],
                B[0, 0] * B[1, 1] - B[0, 1] * B[1, 0],
            ],
        ]
    )
    adj_B_sq_norm = np.trace(adj_B @ adj_B.T)
    det_B = np.linalg.det(B)

    L = np.sum(weights)  # Initial guess is total weight
    for _ in range(max_iter):
        f = (L**2 - B_sq_norm) ** 2 - 8 * L * det_B - 4 * adj_B_sq_norm
        df = 4 * L * (L**2 - B_sq_norm) - 8 * det_B
        dL = f / df
        L -= dL
        if abs(dL) < tol:
            break

    # Compute optimal quaternion via Gibbs vector p
    # q = [p, 1] / sqrt(1 + |p|^2)
    # [(L + sigma)*I - S] * p = Z
    M = (L + sigma) * np.eye(3) - S
    p = np.linalg.solve(M, Z)

    q_opt = np.array([p[0], p[1], p[2], 1.0])
    return quat_normalize(q_opt)
