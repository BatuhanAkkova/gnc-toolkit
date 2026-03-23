"""
Conjunction Analysis and Probability of Collision (Pc) computation.
"""

import numpy as np
from scipy.integrate import dblquad
from scipy.linalg import inv, det

def compute_pc_foster(r1: np.ndarray, v1: np.ndarray, cov1: np.ndarray,
                      r2: np.ndarray, v2: np.ndarray, cov2: np.ndarray,
                      hbr: float) -> float:
    """
    Computes the Probability of Collision (Pc) using Foster's method.
    Assumes Gaussian distribution for states.
    Projects covariance matrices into the collision plane perpendicular to relative velocity.

    Args:
        r1 (np.ndarray): Position of object 1 [m], shape (3,)
        v1 (np.ndarray): Velocity of object 1 [m/s], shape (3,)
        cov1 (np.ndarray): Covariance matrix of object 1 [m^2], shape (3,3)
        r2 (np.ndarray): Position of object 2 [m], shape (3,)
        v2 (np.ndarray): Velocity of object 2 [m/s], shape (3,)
        cov2 (np.ndarray): Covariance matrix of object 2 [m^2], shape (3,3)
        hbr (float): Hard Body Radius [m]

    Returns:
        float: Probability of collision (0 to 1).
    """
    # 1. Relative State at Encounter
    r_rel = r1 - r2
    v_rel = v1 - v2
    v_mag = np.linalg.norm(v_rel)
    
    if v_mag < 1e-6:
        raise ValueError("Relative velocity is too small for encounter plane projection.")

    # 2. Combined Covariance
    cov_comb = cov1 + cov2

    # 3. Encounter Frame Definition (Collision Plane)
    # z-axis along relative velocity
    z_axis = v_rel / v_mag
    
    # x-axis and y-axis spanning the plane perpendicular to z
    # At TCA, r_rel is perpendicular to v_rel. If not, use cross products.
    if np.abs(np.dot(r_rel, z_axis)) > 1e-3:
        # Standardize encounter plane
         x_axis = np.cross(z_axis, [1, 0, 0])
         if np.linalg.norm(x_axis) < 1e-6:
             x_axis = np.cross(z_axis, [0, 1, 0])
         x_axis = x_axis / np.linalg.norm(x_axis)
    else:
        # If at TCA, r_rel is already in the plane
        if np.linalg.norm(r_rel) > 1e-6:
             x_axis = r_rel / np.linalg.norm(r_rel)
        else:
             # Identical position collision
             x_axis = np.array([1, 0, 0]) - z_axis * z_axis[0]
             if np.linalg.norm(x_axis) < 1e-6:
                  x_axis = np.array([0, 1, 0]) - z_axis * z_axis[1]
             x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Rotation matrix from Reference to Encounter Frame (R_E_R)
    R = np.vstack([x_axis, y_axis, z_axis])

    # 4. Project relative position and covariance to encounter plane (2D)
    r_encounter = R @ r_rel
    x_c = r_encounter[0]
    y_c = r_encounter[1]

    # Project covariance
    cov_encounter = R @ cov_comb @ R.T
    cov_2d = cov_encounter[:2, :2]

    # 5. Integration over the Hard Body Circle (Radius HBR)
    det_cov = det(cov_2d)
    if det_cov < 1e-12:
        return 0.0 # Singular covariance
        
    inv_cov = inv(cov_2d)

    def integrand(x, y):
        dx = x - x_c
        dy = y - y_c
        vec = np.array([dx, dy])
        # exponent: -0.5 * vec^T * inv_cov * vec
        exp_val = -0.5 * np.dot(vec, np.dot(inv_cov, vec))
        return (1.0 / (2 * np.pi * np.sqrt(det_cov))) * np.exp(exp_val)

    # Limits of integration:
    # x goes from -sqrt(hbr^2 - y^2) to sqrt(hbr^2 - y^2)
    # y goes from -hbr to hbr
    val, err = dblquad(integrand, -hbr, hbr, 
                       lambda y: -np.sqrt(np.maximum(0, hbr**2 - y**2)), 
                       lambda y: np.sqrt(np.maximum(0, hbr**2 - y**2)))

    return val

def compute_pc_chan(r1: np.ndarray, v1: np.ndarray, cov1: np.ndarray,
                    r2: np.ndarray, v2: np.ndarray, cov2: np.ndarray,
                    hbr: float) -> float:
    """
    Computes Probability of Collision using Chan's analytical approximation.
    Usually faster and applicable when HBR is small relative to state errors.
    """
    # 1. Relative State at Encounter
    r_rel = r1 - r2
    v_rel = v1 - v2
    v_mag = np.linalg.norm(v_rel)
    
    if v_mag < 1e-6:
        raise ValueError("Relative velocity is too small for encounter plane projection.")

    # 2. Combined Covariance
    cov_comb = cov1 + cov2

    # 3. Encounter Frame Definition (Collision Plane)
    z_axis = v_rel / v_mag
    
    if np.abs(np.dot(r_rel, z_axis)) > 1e-3:
         x_axis = np.cross(z_axis, [1, 0, 0])
         if np.linalg.norm(x_axis) < 1e-6:
             x_axis = np.cross(z_axis, [0, 1, 0])
         x_axis = x_axis / np.linalg.norm(x_axis)
    else:
        if np.linalg.norm(r_rel) > 1e-6:
             x_axis = r_rel / np.linalg.norm(r_rel)
        else:
             x_axis = np.array([1, 0, 0]) - z_axis * z_axis[0]
             if np.linalg.norm(x_axis) < 1e-6:
                  x_axis = np.array([0, 1, 0]) - z_axis * z_axis[1]
             x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    R = np.vstack([x_axis, y_axis, z_axis])

    # 4. Project relative position and covariance to encounter plane (2D)
    r_encounter = R @ r_rel
    cov_encounter = R @ cov_comb @ R.T
    cov_2d = cov_encounter[:2, :2]

    # Diagonalize cov_2d to find principal axes
    eigvals, eigvecs = np.linalg.eigh(cov_2d)
    
    # Check for singular covariance
    if np.any(eigvals <= 0) or np.prod(eigvals) < 1e-12:
        return 0.0

    # Rotate relative position to principal axes
    r_principal = eigvecs.T @ r_encounter[:2]
    x_c = r_principal[0]
    y_c = r_principal[1]
    
    var_x = eigvals[0]
    var_y = eigvals[1]

    # Chan variables
    u = (x_c**2 / var_x) + (y_c**2 / var_y)
    v = (hbr**2) / np.sqrt(var_x * var_y)

    A = u / 2.0
    B = v / 2.0

    max_terms = 100
    tol = 1e-12

    try:
        exp_A = np.exp(-A)
        exp_B = np.exp(-B)
    except:
        return 0.0

    if exp_A < 1e-100:
        return 0.0
        
    term_A = exp_A
    term_B = exp_B
    sum_B = term_B
    
    pc = 0.0
    
    for n in range(max_terms):
        current_term = term_A * (1.0 - sum_B)
        pc += current_term
        
        if n > A and current_term < tol:
            break
            
        term_A = term_A * A / (n + 1)
        term_B = term_B * B / (n + 1)
        sum_B += term_B

    return pc
