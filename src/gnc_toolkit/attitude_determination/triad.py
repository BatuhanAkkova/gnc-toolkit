import numpy as np

def triad(v_body1, v_body2, v_ref1, v_ref2):
    """
    Compute the rotation matrix (Direction Cosine Matrix) from Inertial to Body frame
    using the TRIAD algorithm.

    Args:
        v_body1 (np.ndarray): First vector measured in body frame (e.g. sun vector).
        v_body2 (np.ndarray): Second vector measured in body frame (e.g. mag vector).
        v_ref1 (np.ndarray): First vector in reference/inertial frame (corresponding to v_body1).
        v_ref2 (np.ndarray): Second vector in reference/inertial frame (corresponding to v_body2).

    Returns:
        np.ndarray: 3x3 Rotation matrix R_BI (Ref -> Body). Puts vectors in Body frame.
                    v_body = R_BI @ v_ref
    """
    # Normalize inputs
    b1 = v_body1 / np.linalg.norm(v_body1)
    b2 = v_body2 / np.linalg.norm(v_body2)
    r1 = v_ref1 / np.linalg.norm(v_ref1)
    r2 = v_ref2 / np.linalg.norm(v_ref2)

    # Construct Body Triad
    t1b = b1
    t2b_raw = np.cross(b1, b2)
    norm_t2b = np.linalg.norm(t2b_raw)
    
    if norm_t2b < 1e-8:
        raise ValueError("Body vectors are collinear or nearly collinear.")
    
    t2b = t2b_raw / norm_t2b
    t3b = np.cross(t1b, t2b)

    # Construct Reference Triad
    t1r = r1
    t2r_raw = np.cross(r1, r2)
    norm_t2r = np.linalg.norm(t2r_raw)
    
    if norm_t2r < 1e-8:
        raise ValueError("Reference vectors are collinear or nearly collinear.")
        
    t2r = t2r_raw / norm_t2r
    t3r = np.cross(t1r, t2r)

    # Construct DCMs
    # M_b = [t1b t2b t3b]
    # M_r = [t1r t2r t3r]
    # R_BI * M_r = M_b  =>  R_BI = M_b * M_r^T
    
    M_b = np.column_stack((t1b, t2b, t3b))
    M_r = np.column_stack((t1r, t2r, t3r))

    R_BI = M_b @ M_r.T
    
    return R_BI
