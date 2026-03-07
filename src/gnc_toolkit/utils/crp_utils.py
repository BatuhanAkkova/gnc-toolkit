import numpy as np

def quat_to_crp(q):
    """
    Convert quaternion [x, y, z, w] to Classical Rodrigues Parameters (CRP/Gibbs).
    
    q_vec / q_w
    """
    x, y, z, w = q
    if abs(w) < 1e-12:
        raise ValueError("Classical Rodrigues Parameters are singular for 180-degree rotations.")
    return np.array([x, y, z]) / w

def crp_to_quat(q_crp):
    """
    Convert CRP to quaternion [x, y, z, w].
    
    q_w = 1 / sqrt(1 + |q_crp|^2)
    q_vec = q_crp / sqrt(1 + |q_crp|^2)
    """
    norm_sq = np.sum(q_crp**2)
    den = np.sqrt(1 + norm_sq)
    return np.array([q_crp[0]/den, q_crp[1]/den, q_crp[2]/den, 1.0/den])

def crp_to_dcm(q_crp):
    """
    Convert CRP to 3x3 Direction Cosine Matrix.
    """
    norm_sq = np.sum(q_crp**2)
    q1, q2, q3 = q_crp
    
    # R = I + [2*(S^2 - S)] / (1 + |q_crp|^2)
    # where S is skew-symmetric matrix.
    S = np.array([
        [0, -q3, q2],
        [q3, 0, -q1],
        [-q2, q1, 0]
    ])
    
    I = np.eye(3)
    R = I + (2 * S @ S - 2 * S) / (1 + norm_sq)
    return R

def crp_addition(q1, q2):
    """
    Perform CRP addition (rotation composition).
    
    q_res = [q1 + q2 + q2 x q1] / (1 - q1 . q2)
    """
    dot = np.dot(q1, q2)
    if abs(1 - dot) < 1e-12:
        raise ValueError("CRP addition is singular.")
        
    cross = np.cross(q2, q1)
    return (q1 + q2 + cross) / (1 - dot)
