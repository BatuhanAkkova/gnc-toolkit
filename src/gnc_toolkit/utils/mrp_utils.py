"""
Modified Rodrigues Parameters (MRP) kinematics and math utilities.
"""

import numpy as np
from gnc_toolkit.utils.quat_utils import quat_normalize

def quat_to_mrp(q):
    """
    Convert quaternion [x, y, z, w] to Modified Rodrigues Parameters (MRP).
    
    sigma = q_vec / (1 + q_w)
    """
    x, y, z, w = q
    return np.array([x, y, z]) / (1 + w)

def mrp_to_quat(sigma):
    """
    Convert Modified Rodrigues Parameters (MRP) to quaternion [x, y, z, w].
    
    q_w = (1 - |sigma|^2) / (1 + |sigma|^2)
    q_vec = 2 * sigma / (1 + |sigma|^2)
    """
    sigma_sq = np.sum(sigma**2)
    w = (1 - sigma_sq) / (1 + sigma_sq)
    xyz = 2 * sigma / (1 + sigma_sq)
    return np.array([xyz[0], xyz[1], xyz[2], w])

def mrp_to_dcm(sigma):
    """
    Convert MRP to 3x3 Direction Cosine Matrix.
    """
    sigma_sq = np.sum(sigma**2)
    s_x = sigma[0]
    s_y = sigma[1]
    s_z = sigma[2]
    
    S = np.array([
        [0, -s_z, s_y],
        [s_z, 0, -s_x],
        [-s_y, s_x, 0]
    ])
    
    I = np.eye(3)
    R = I + (8 * S @ S - 4 * (1 - sigma_sq) * S) / (1 + sigma_sq)**2
    return R

def get_shadow_mrp(sigma):
    """
    Return the shadow set for the given MRP.
    
    sigma_shadow = -sigma / |sigma|^2
    """
    sigma_sq = np.sum(sigma**2)
    if sigma_sq < 1e-12:
        return np.zeros(3)
    return -sigma / sigma_sq

def check_mrp_switching(sigma, threshold=1.0):
    """
    Check if the MRP should be switched to its shadow set to avoid singularity.
    
    Args:
        sigma (np.ndarray): Current MRP.
        threshold (float): Switching threshold (typically 1.0).
        
    Returns:
        np.ndarray: Switched (or original) MRP.
    """
    if np.linalg.norm(sigma) > threshold:
        return get_shadow_mrp(sigma)
    return sigma
