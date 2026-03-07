import numpy as np

def foam(body_vectors, ref_vectors, weights=None, tol=1e-12, max_iter=20):
    """
    Compute the optimal attitude matrix (Direction Cosine Matrix) using the FOAM algorithm.
    
    FOAM (Fast Optimal Attitude Matrix) solves the Wahba problem by directly 
    computing the DCM without intermediate quaternion representations.

    Args:
        body_vectors (list or np.ndarray): List of N vectors measured in body frame. Shape (N, 3).
        ref_vectors (list or np.ndarray): List of N vectors known in reference (inertial) frame. Shape (N, 3).
        weights (list or np.ndarray, optional): List of N scalar weights. If None, weights are equal.
        tol (float, optional): Convergence tolerance for the iterative eigenvalue search.
        max_iter (int, optional): Maximum number of iterations for eigenvalue search.
                                       
    Returns:
        np.ndarray: 3x3 Direction Cosine Matrix R_BI (Ref -> Body).
                    v_body = R_BI @ v_ref
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

    # Normalize input vectors
    b_vecs_norm = b_vecs / np.linalg.norm(b_vecs, axis=1)[:, np.newaxis]
    r_vecs_norm = r_vecs / np.linalg.norm(r_vecs, axis=1)[:, np.newaxis]

    # Compute Attitude Profile Matrix B = sum(w_i * b_i * r_i^T)
    B = np.zeros((3, 3))
    for i in range(n):
        B += weights[i] * np.outer(b_vecs_norm[i], r_vecs_norm[i])

    det_B = np.linalg.det(B)
    adj_B = np.linalg.det(B) * np.linalg.inv(B).T if det_B != 0 else np.zeros((3, 3)) # Adjugate of B
    # Correct adjugate calculation for small det cases (manual cofactor matrix)
    if det_B == 0:
        adj_B = np.array([
            [B[1,1]*B[2,2]-B[1,2]*B[2,1], B[0,2]*B[2,1]-B[0,1]*B[2,2], B[0,1]*B[1,2]-B[0,2]*B[1,1]],
            [B[1,2]*B[2,0]-B[1,0]*B[2,2], B[0,0]*B[2,2]-B[0,2]*B[2,0], B[0,2]*B[1,0]-B[0,0]*B[1,2]],
            [B[1,0]*B[2,1]-B[1,1]*B[2,0], B[0,1]*B[2,0]-B[0,0]*B[2,1], B[0,0]*B[1,1]-B[0,1]*B[1,0]]
        ])

    B_sq_norm = np.trace(B @ B.T)
    adj_B_sq_norm = np.trace(adj_B @ adj_B.T)

    # Solve for lambda (maximum eigenvalue of K) using Newton-Raphson
    # Characteristic polynomial: f(L) = (L^2 - B_sq_norm)^2 - 8*L*det(B) - 4*adj_B_sq_norm
    # Note: FOAM uses a slightly different form but equivalent to QUEST's max eigenvalue goal.
    
    L = np.sum(weights) # Initial guess
    for _ in range(max_iter):
        f = (L**2 - B_sq_norm)**2 - 8*L*det_B - 4*adj_B_sq_norm
        df = 4*L*(L**2 - B_sq_norm) - 8*det_B
        
        dL = f / df
        L -= dL
        if abs(dL) < tol:
            break

    # DCM calculation: R = ((L^2 + B_sq_norm)*B + 2*L*adj_B.T - B * B.T * B) / (L*(L^2 - B_sq_norm) - 2*det_B)
    # Using the simplified FOAM DCM expression:
    kappa = 0.5 * (L**2 - B_sq_norm)
    zeta = L * det_B - adj_B_sq_norm # This is not standard, let's use the robust form
    
    # Robust DCM from Markley:
    # R = ((L^2 + B_sq_norm)*B + 2*L*adj_B.T - 2*B @ B.T @ B) / (L*(L^2 - B_sq_norm) - 2*det_B)
    # Actually, the standard FOAM DCM is:
    # R = [ (L^2 + B_sq_norm)*B + 2*L*adj(B)^T - 2*B*B^T*B ] / [ L(L^2 - B_sq_norm) - 2*det(B) ]
    
    num = (L**2 + B_sq_norm) * B + 2 * L * adj_B.T - 2 * (B @ B.T @ B)
    den = L * (L**2 - B_sq_norm) - 2 * det_B
    
    R_BI = num / den
    
    return R_BI
