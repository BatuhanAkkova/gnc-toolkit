"""
Euler angle kinematics and Direction Cosine Matrix (DCM) utilities.
"""

import numpy as np
from gnc_toolkit.utils.state_conversion import rot_x, rot_y, rot_z

def euler_to_dcm(angles, sequence):
    """
    Convert Euler angles [theta1, theta2, theta3] to 3x3 Direction Cosine Matrix (Body to ECI).
    
    Standard convention for Body-to-ECI is R = R3*R2*R1 where R1 is the first rotation.

    Args:
        angles (list/np.ndarray): Three Euler angles in radians.
        sequence (str): Sequence of rotations (e.g. '321', '313').
        
    Returns:
        np.ndarray: 3x3 Direction Cosine Matrix R_BI (Ref -> Body).
    """
    if len(sequence) != 3:
        raise ValueError("Sequence must be a string of length 3.")
        
    rot_fns = {'1': rot_x, '2': rot_y, '3': rot_z}
    
    R1 = rot_fns[sequence[0]](angles[0])
    R2 = rot_fns[sequence[1]](angles[1])
    R3 = rot_fns[sequence[2]](angles[2])
    
    return R3 @ R2 @ R1

def dcm_to_euler(dcm, sequence):
    """
    Convert 3x3 Direction Cosine Matrix to Euler angles for any of the 12 sequences.
    
    Args:
        dcm (np.ndarray): 3x3 Direction Cosine Matrix.
        sequence (str): Sequence of rotations (e.g. '321', '313').
        
    Returns:
        np.ndarray: [theta1, theta2, theta3] in radians.
    """
    # Symmetric Sequences (e.g., 3-1-3)
    if sequence[0] == sequence[2]:
        i = int(sequence[0]) - 1
        j = int(sequence[1]) - 1
        k = 3 - i - j
        
        # Determine sign parity
        parity = 1 if (j - i) % 3 == 1 else -1
        
        theta2 = np.arccos(np.clip(dcm[i, i], -1, 1))
        
        if abs(np.sin(theta2)) < 1e-12:
            # Singularity (theta2 = 0 or PI)
            theta1 = 0.0
            theta3 = np.arctan2(parity * dcm[j, k], dcm[j, j])
        else:
            theta1 = np.arctan2(dcm[j, i], -parity * dcm[k, i])
            theta3 = np.arctan2(dcm[i, j], parity * dcm[i, k])
            
        return np.array([theta1, theta2, theta3])

    # Asymmetric Sequences (e.g., 3-2-1)
    else:
        i = int(sequence[0]) - 1
        j = int(sequence[1]) - 1
        k = int(sequence[2]) - 1
        
        # Determine sign parity
        parity = 1 if (j - i) % 3 == 1 else -1
        
        theta2 = np.arcsin(np.clip(-parity * dcm[i, k], -1, 1))
        
        if abs(np.cos(theta2)) < 1e-12:
            # Singularity (theta2 = PI/2)
            theta1 = 0.0
            theta3 = np.arctan2(parity * dcm[j, i], dcm[j, j])
        else:
            theta1 = np.arctan2(parity * dcm[j, k], dcm[k, k])
            theta3 = np.arctan2(parity * dcm[i, j], dcm[i, i])
            
        return np.array([theta1, theta2, theta3])
