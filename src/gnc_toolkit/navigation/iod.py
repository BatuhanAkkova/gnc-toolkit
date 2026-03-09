import numpy as np

def gibbs_iod(r1, r2, r3, mu=398600.4415e9):
    """
    Gibbs method for Initial Orbit Determination (3 position vectors).
    Used for long-arc observations (separation > 5 degrees).
    
    Args:
        r1, r2, r3 (np.ndarray): Three position vectors in ECI [m]
        mu (float): Gravitational parameter
        
    Returns:
        v2 (np.ndarray): Velocity vector at time t2 [m/s]
    """
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    r3_mag = np.linalg.norm(r3)
    
    # Check coplanarity
    h_hat = np.cross(r1, r2)
    h_hat /= np.linalg.norm(h_hat)
    if abs(np.dot(r3, h_hat)) > 1e-4 * r3_mag:
        # Not perfectly coplanar, but we proceed
        pass

    # Gibbs vectors
    D = np.cross(r1, r2) + np.cross(r2, r3) + np.cross(r3, r1)
    N = r1_mag * np.cross(r2, r3) + r2_mag * np.cross(r3, r1) + r3_mag * np.cross(r1, r2)
    L = np.linalg.norm(D)
    
    if L < 1e-12:
        return np.zeros(3)
        
    v2 = np.sqrt(mu / (np.linalg.norm(N) * L)) * (np.cross(D, r2) / r2_mag + L * np.array([0,0,0])) # Placeholder for logic
    # Re-evaluating v2 formula: v2 = sqrt(mu/(N_mag*D_mag)) * [ (D x r2)/r2_mag + N ]
    v2 = np.sqrt(mu / (np.linalg.norm(N) * L)) * (np.cross(D, r2) / r2_mag + N)
    
    return v2

def herrick_gibbs_iod(r1, r2, r3, dt21, dt32, mu=398600.4415e9):
    """
    Herrick-Gibbs method for IOD (3 position vectors, short arc).
    Used when separation between vectors is small (< 5-10 degrees).
    
    Args:
        r1, r2, r3 (np.ndarray): Position vectors
        dt21 (float): t2 - t1
        dt32 (float): t3 - t2
        mu (float): Gravitational parameter
        
    Returns:
        v2 (np.ndarray): Velocity at t2
    """
    dt31 = dt21 + dt32
    
    term1 = -dt32 * (1.0/(dt21*dt31) + mu/(12.0 * np.linalg.norm(r1)**3)) * r1
    term2 = (dt32 - dt21) * (1.0/(dt21*dt32) + mu/(12.0 * np.linalg.norm(r2)**3)) * r2
    term3 = dt21 * (1.0/(dt32*dt31) + mu/(12.0 * np.linalg.norm(r3)**3)) * r3
    
    v2 = term1 + term2 + term3
    return v2

def gauss_iod(rho_hat1, rho_hat2, rho_hat3, t1, t2, t3, R1, R2, R3, mu=398600.4415e9):
    """
    Gauss method for Initial Orbit Determination (3 line-of-sight vectors).
    rho_hat: unit line-of-sight from observer
    Ri: observer position in ECI
    """
    tau1 = t1 - t2
    tau3 = t3 - t2
    tau = tau3 - tau1
    
    # Scalar products
    D11 = np.dot(np.cross(rho_hat1, rho_hat2), rho_hat3)
    # This involves solving an 8th order polynomial (Gauss's polynomial)
    # Simplified implementation placeholder
    
    # In practice, this needs an iterative solver for rho2_mag
    # We will implement the core logic for the 8th order equation in a full version.
    
    # For now, return a placeholder.
    return np.zeros(6)

def laplace_iod(rho_hat, rho_hat_dot, rho_hat_ddot, R, R_dot, R_ddot, mu=398600.4415e9):
    """
    Laplace method for IOD using LOS derivatives at a single epoch.
    
    Args:
        rho_hat (np.ndarray): Unit LOS vector
        rho_hat_dot (np.ndarray): First derivative of unit LOS
        rho_hat_ddot (np.ndarray): Second derivative of unit LOS
        R (np.ndarray): Observer position vector [m]
        R_dot (np.ndarray): Observer velocity vector [m/s]
        R_ddot (np.ndarray): Observer acceleration vector [m/s^2]
        mu (float): Gravitational parameter
        
    Returns:
        np.ndarray: [rx, ry, rz, vx, vy, vz] State vector in ECI [m, m/s]
    """
    # Determinants for Laplace method (Standard textbook definition)
    D = np.linalg.det(np.array([rho_hat, rho_hat_dot, rho_hat_ddot]))
    
    if abs(D) < 1e-18:
        raise ValueError("Determinant D is too small; observations may be nearly coplanar or insufficient.")
        
    D1 = np.linalg.det(np.array([rho_hat, rho_hat_dot, R_ddot]))
    D2 = np.linalg.det(np.array([rho_hat, rho_hat_dot, R]))
    
    A = -D1 / D
    B = -mu * D2 / D
    
    R_mag = np.linalg.norm(R)
    cos_phi = np.dot(rho_hat, R) / R_mag
    
    # Solve 8th order polynomial for r (radius of satellite)
    # r^8 - (A^2 + 2*R*A*cos_phi + R^2)*r^6 - (2*A*B + 2*R*B*cos_phi)*r^3 - B^2 = 0
    
    poly = [
        1.0,                               # r^8
        0.0,                               # r^7
        -(A**2 + 2.0 * R_mag * A * cos_phi + R_mag**2), # r^6
        0.0,                               # r^5
        0.0,                               # r^4
        -(2.0 * A * B + 2.0 * R_mag * B * cos_phi),    # r^3
        0.0,                               # r^2
        0.0,                               # r^1
        -B**2                              # r^0
    ]
    
    roots = np.roots(poly)
    # We need the positive real root
    real_positive_roots = roots[np.isreal(roots) & (roots.real > 0)].real
    if len(real_positive_roots) == 0:
        raise ValueError("No physical (positive real) root found for Laplace IOD.")
        
    # Usually choose root closest to earth-radius for LEO, but pick the first positive one
    r_mag = real_positive_roots[0]
    
    rho = A + B / r_mag**3
    
    # Position
    r_vec = rho * rho_hat + R
    
    # Velocity
    # rho_dot = -(D3 + (mu/r^3)*D4) / (2*D)  -- Note the 2 in the divisor!
    D3 = np.linalg.det(np.array([rho_hat, R_ddot, rho_hat_ddot]))
    D4 = np.linalg.det(np.array([rho_hat, R, rho_hat_ddot]))
    rho_dot = -(D3 + (mu / r_mag**3) * D4) / (2.0 * D)
    
    v_vec = rho_dot * rho_hat + rho * rho_hat_dot + R_dot
    
    return np.concatenate([r_vec, v_vec])

def laplace_iod_from_observations(rho_hats, Rs, times, mu=398600.4415e9):
    """
    Helper to perform Laplace IOD from three LOS observations.
    Estimates derivatives using Lagrange interpolation at the middle epoch.
    
    Args:
        rho_hats (list of np.ndarray): 3 LOS unit vectors
        Rs (list of np.ndarray): 3 Observer position vectors [m]
        times (list of float): 3 observation timestamps [s]
    """
    t1, t2, t3 = times
    L1, L2, L3 = rho_hats
    R1, R2, R3 = Rs
    
    # Time intervals
    tau32 = t3 - t2
    tau21 = t2 - t1
    tau31 = t3 - t1
    
    # Lagrange interpolation coefficients for derivatives at t2
    # L(t) = L1*l1(t) + L2*l2(t) + L3*l3(t)
    # l1'(t2) = (t2-t3) / ((t1-t2)(t1-t3))
    # l2'(t2) = (2*t2 - t1 - t3) / ((t2-t1)(t2-t3))
    # l3'(t2) = (t2-t1) / ((t3-t1)(t3-t2))
    
    l1_dot = -tau32 / (-tau21 * -tau31)
    l2_dot = (tau21 - tau32) / (tau21 * -tau32)
    l3_dot = tau21 / (tau31 * tau32)
    
    rho_hat_dot = l1_dot * L1 + l2_dot * L2 + l3_dot * L3
    R_dot = l1_dot * R1 + l2_dot * R2 + l3_dot * R3
    
    # Second derivatives at t2
    # l1''(t2) = 2 / ((t1-t2)(t1-t3))
    # l2''(t2) = 2 / ((t2-t1)(t2-t3))
    # l3''(t2) = 2 / ((t3-t1)(t3-t2))
    
    l1_ddot = 2.0 / (-tau21 * -tau31)
    l2_ddot = 2.0 / (tau21 * -tau32)
    l3_ddot = 2.0 / (tau31 * tau32)
    
    rho_hat_ddot = l1_ddot * L1 + l2_ddot * L2 + l3_ddot * L3
    
    # Observer acceleration estimation (or use gravity if R is ECI)
    # For ground stations, we can differentiate. 
    # To be consistent with LOS interpolation, we differentiate R.
    R_ddot = l1_ddot * R1 + l2_ddot * R2 + l3_ddot * R3
    
    return laplace_iod(L2, rho_hat_dot, rho_hat_ddot, R2, R_dot, R_ddot, mu)
