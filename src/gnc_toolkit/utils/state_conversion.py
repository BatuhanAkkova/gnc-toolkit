import numpy as np

def quat_to_dcm(q):
    """Convert quaternion [x, y, z, w] to Direction Cosine Matrix (Body to ECI)."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
    ])

def quat_to_euler(q, sequence):
    """Convert quaternion [x, y, z, w] to Euler angles in any sequence."""
    if len(sequence) != 3:
        raise ValueError("Sequence must be a string of length 3.")
    dcm = quat_to_dcm(q)
    return dcm_to_euler(dcm, sequence)

def dcm_to_quat(dcm):
    """Convert Direction Cosine Matrix (Body to ECI) to quaternion [x, y, z, w]."""
    # Shepperd's algorithm or simple trace check
    tr = np.trace(dcm)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2 
        w = 0.25 * S
        x = (dcm[2, 1] - dcm[1, 2]) / S
        y = (dcm[0, 2] - dcm[2, 0]) / S
        z = (dcm[1, 0] - dcm[0, 1]) / S
    elif (dcm[0, 0] > dcm[1, 1]) and (dcm[0, 0] > dcm[2, 2]):
        S = np.sqrt(1.0 + dcm[0, 0] - dcm[1, 1] - dcm[2, 2]) * 2
        w = (dcm[2, 1] - dcm[1, 2]) / S
        x = 0.25 * S
        y = (dcm[0, 1] + dcm[1, 0]) / S
        z = (dcm[0, 2] + dcm[2, 0]) / S
    elif dcm[1, 1] > dcm[2, 2]:
        S = np.sqrt(1.0 + dcm[1, 1] - dcm[0, 0] - dcm[2, 2]) * 2
        w = (dcm[0, 2] - dcm[2, 0]) / S
        x = (dcm[0, 1] + dcm[1, 0]) / S
        y = 0.25 * S
        z = (dcm[1, 2] + dcm[2, 1]) / S
    else:
        S = np.sqrt(1.0 + dcm[2, 2] - dcm[0, 0] - dcm[1, 1]) * 2
        w = (dcm[1, 0] - dcm[0, 1]) / S
        x = (dcm[0, 2] + dcm[2, 0]) / S
        y = (dcm[1, 2] + dcm[2, 1]) / S
        z = 0.25 * S
    return np.array([x, y, z, w])

def dcm_to_euler(dcm, sequence):
    """Convert Direction Cosine Matrix (Body to ECI) to Euler angles in any sequence."""
    if len(sequence) != 3:
        raise ValueError("Sequence must be a string of length 3.")
    
    if sequence == "123":
        theta2 = -np.arcsin(np.clip(dcm[2, 0], -1, 1))
        theta1 = np.arctan2(dcm[2, 1], dcm[2, 2])
        theta3 = np.arctan2(dcm[1, 0], dcm[0, 0])
        return np.array([theta1, theta2, theta3])
        
    elif sequence == "321":
        theta2 = np.arcsin(np.clip(dcm[0, 2], -1, 1))
        theta1 = np.arctan2(-dcm[0, 1], dcm[0, 0])
        theta3 = np.arctan2(-dcm[1, 2], dcm[2, 2])
        return np.array([theta1, theta2, theta3])
    
    else:
        raise NotImplementedError("Only sequences 123 and 321 are fully supported/verified in this patch.")

def euler_to_quat(angle, sequence):
    """Convert Euler angles in any sequence to quaternion [x, y, z, w]."""
    if len(sequence) != 3:
        raise ValueError("Sequence must be a string of length 3.")
    dcm = euler_to_dcm(angle, sequence)
    return dcm_to_quat(dcm)

def euler_to_dcm(angle, sequence):
    """Convert Euler angles in any sequence to Direction Cosine Matrix (Body to ECI)."""
    if len(sequence) != 3:
        raise ValueError("Sequence must be a string of length 3.")
    rotations = []
    for i, char in enumerate(sequence):
        axis = int(char)
        theta = angle[i]
        if axis == 1:
            R = rot_x(theta)
        elif axis == 2:
            R = rot_y(theta)
        elif axis == 3:
            R = rot_z(theta)
        rotations.append(R)
    return rotations[2] @ rotations[1] @ rotations[0]

def rot_x(angle):
    """Rotation matrix for rotation about x-axis."""
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])

def rot_y(angle):
    """Rotation matrix for rotation about y-axis."""
    return np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])

def rot_z(angle):
    """Rotation matrix for rotation about z-axis."""
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
