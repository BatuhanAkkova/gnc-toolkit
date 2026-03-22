"""
Continuous thrust guidance laws including Q-law and ZEM/ZEV feedback.
"""

import numpy as np
from gnc_toolkit.utils.state_to_elements import eci2kepler

def q_law_guidance(r: np.ndarray, v: np.ndarray, target_oe: np.ndarray, mu: float, f_max: float, weights: np.ndarray = None) -> np.ndarray:
    """
    Lyapunov-based Q-law guidance for low-thrust orbit transfers.
    Targets [a, e, i, raan, argp].
    
    Args:
        r (np.ndarray): Position ECI (m).
        v (np.ndarray): Velocity ECI (m/s).
        target_oe (np.ndarray): Target Keplerian elements [a, e, i, raan, argp] (m, rad).
        mu (float): Gravitational parameter (m^3/s^2).
        f_max (float): Maximum thrust acceleration (m/s^2).
        weights (np.ndarray): Weights for each element in Q function.
        
    Returns:
        np.ndarray: Optimal thrust acceleration vector in ECI (m/s^2).
    """
    a, e, i, raan, argp, nu, M, E, p, arglat, truelon, lonper = eci2kepler(r, v)
    
    oe = np.array([a, e, i, raan, argp])
    oe_target = target_oe[:5]
    
    if weights is None:
        weights = np.array([1.0/a**2, 1.0, 1.0, 1.0, 1.0])
        
    # Gauss Variational Equations (GVE) in RTN frame
    r_mag = np.linalg.norm(r)
    h_vec = np.cross(r, v)
    h = np.linalg.norm(h_vec)
    theta = argp + nu
    
    # B_a (radial, tangential, normal)
    B_a = (2 * a**2 / h) * np.array([e * np.sin(nu), p/r_mag, 0.0])
    
    # B_e
    B_e = (1/h) * np.array([p * np.sin(nu), (p+r_mag)*np.cos(nu) + r_mag*e, 0.0])
    
    # B_i
    B_i = (r_mag * np.cos(theta) / h) * np.array([0.0, 0.0, 1.0])
    
    # B_raan
    if np.sin(i) == 0:
        B_raan = np.zeros(3)
    else:
        B_raan = (r_mag * np.sin(theta) / (h * np.sin(i))) * np.array([0.0, 0.0, 1.0])
        
    # B_argp
    term1 = (1 / (h*e)) * np.array([-p*np.cos(nu), (p+r_mag)*np.sin(nu), 0.0])
    if np.sin(i) == 0:
        term2 = np.zeros(3)
    else:
        term2 = (r_mag * np.sin(theta) * np.cos(i) / (h * np.sin(i))) * np.array([0.0, 0.0, 1.0])
    B_argp = term1 - term2
    
    B = np.vstack([B_a, B_e, B_i, B_raan, B_argp])
    
    # Gradient of Q = sum W_i * (oe_i - target_i)^2
    # dQ/doe = 2 * W_i * (oe_i - target_i)
    grad_Q_oe = 2 * weights * (oe - oe_target)
    
    # Optimal thrust direction in RTN is anti-parallel to grad_Q_oe * B
    # Direction D = -(grad_Q_oe @ B)
    D = - (grad_Q_oe @ B)
    
    if np.linalg.norm(D) < 1e-12:
        return np.zeros(3)
        
    u_rtn = D / np.linalg.norm(D)
    f_rtn = f_max * u_rtn
    
    # Convert RTN to ECI
    u_r = r / r_mag
    u_n = h_vec / h
    u_t = np.cross(u_n, u_r)
    
    R_rtn_to_eci = np.column_stack([u_r, u_t, u_n])
    f_eci = R_rtn_to_eci @ f_rtn
    
    return f_eci

def zem_zev_guidance(r: np.ndarray, v: np.ndarray, r_target: np.ndarray, v_target: np.ndarray, t_go: float, gravity: np.ndarray = None) -> np.ndarray:
    """
    Zero-Effort Miss (ZEM) and Zero-Effort Velocity (ZEV) feedback guidance.
    Used for terminal intercept or landing.
    
    Args:
        r (np.ndarray): Current position (m).
        v (np.ndarray): Current velocity (m/s).
        r_target (np.ndarray): Target position (m).
        v_target (np.ndarray): Target velocity (m/s).
        t_go (float): Time-to-go (s).
        gravity (np.ndarray): Local gravity vector (m/s^2). Defaults to zero.
        
    Returns:
        np.ndarray: Commanded acceleration vector (m/s^2).
    """
    if t_go <= 1e-6:
        return np.zeros(3)
        
    if gravity is None:
        gravity = np.zeros(3)
        
    # ZEM = r_target - (r + v * t_go + 0.5 * gravity * t_go^2)
    # ZEV = v_target - (v + gravity * t_go)
    
    zem = r_target - (r + v * t_go + 0.5 * gravity * t_go**2)
    zev = v_target - (v + gravity * t_go)
    
    a_cmd = (6.0 / t_go**2) * zem - (2.0 / t_go) * zev
    
    return a_cmd

def gravity_turn_guidance(v: np.ndarray, f_mag: float, mode: str = 'descent') -> np.ndarray:
    """
    Gravity turn steering law. Thrust is aligned with velocity vector.
    
    Args:
        v (np.ndarray): Velocity vector (m/s).
        f_mag (float): Thrust acceleration magnitude (m/s^2).
        mode (str): 'descent' (anti-parallel to v) or 'ascent' (parallel to v).
        
    Returns:
        np.ndarray: Thrust acceleration vector (m/s^2).
    """
    v_mag = np.linalg.norm(v)
    if v_mag < 1e-6:
        return np.zeros(3)
        
    u_v = v / v_mag
    
    if mode == 'descent':
        return -f_mag * u_v
    else:
        return f_mag * u_v

def apollo_dps_guidance(t_go: float, r: np.ndarray, v: np.ndarray, r_target: np.ndarray, v_target: np.ndarray, a_target: np.ndarray, gravity: np.ndarray = None) -> np.ndarray:
    """
    E-guidance (Apollo Descent Propulsion System style).
    Targets position, velocity, and acceleration.
    
    Args:
        t_go (float): Time-to-go (s).
        r, v: Current state.
        r_target, v_target, a_target: Target state.
        gravity (np.ndarray): Local gravity.
        
    Returns:
        np.ndarray: Commanded acceleration (m/s^2).
    """
    if t_go <= 1e-6:
        return a_target
        
    if gravity is None:
        gravity = np.zeros(3)
        
    # General polynomial guidance
    delta_r = r_target - (r + v * t_go + 0.5 * gravity * t_go**2)
    delta_v = v_target - (v + gravity * t_go)
    
    # Standard Apollo lunar landing coefficients
    a_cmd = a_target + (12.0 / t_go**2) * delta_r + (6.0 / t_go) * delta_v
    
    return a_cmd

from scipy.integrate import solve_bvp

def indirect_optimal_guidance(r0, v0, rf, vf, tf, mu):
    """
    Indirect optimal guidance using Pontryagin's Minimum Principle (PMP).
    Solves for minimum energy (acceleration squared integral).
    This is a Boundary Value Problem (BVP).
    
    Returns:
        tuple: (time_array, acceleration_profile)
    """
    def fun(t, y):
        r = y[:3]
        v = y[3:6]
        lr = y[6:9]
        lv = y[9:12]
        
        r_mag = np.linalg.norm(r, axis=0)
        # Dynamics
        dr = v
        dv = -(mu / r_mag**3) * r - lv
        
        # Costate dynamics: dot(lambda) = -dH/dx
        # Gravity gradient: dg/dr = -(mu/r^3)I + 3(mu/r^5)r*r^T
        r_mag_mat = np.tile(r_mag, (3, 1))
        unit_r = r / r_mag_mat
        
        # G(r) = -(mu/r^3) * (I - 3*u*u^T)
        dlr = np.zeros_like(lr)
        for i in range(r.shape[1]):
            ri = r[:, i]
            rm = r_mag[i]
            lvi = lv[:, i]
            grad_g = -(mu / rm**3) * (np.eye(3) - 3 * np.outer(ri, ri) / rm**2)
            dlr[:, i] = -grad_g @ lvi
            
        dlv = -lr
        
        return np.vstack([dr, dv, dlr, dlv])

    def bc(ya, yb):
        return np.concatenate([
            ya[:3] - r0,
            ya[3:6] - v0,
            yb[:3] - rf,
            yb[3:6] - vf
        ])

    t_init = np.linspace(0, tf, 5)
    y_init = np.zeros((12, t_init.size))
    # Interpolate state for initial guess
    for i in range(3):
        y_init[i] = np.linspace(r0[i], rf[i], t_init.size)
        y_init[i+3] = np.linspace(v0[i], vf[i], t_init.size)

    res = solve_bvp(fun, bc, t_init, y_init)
    
    if res.success:
        lv_sol = res.y[9:12]
        return res.x, -lv_sol
    else:
        return None, None

from scipy.optimize import minimize

def direct_collocation_guidance(r0, v0, rf, vf, tf, mu, n_nodes=20):
    """
    Direct transcription / collocation (Trapezoidal).
    Minimizes sum of acceleration squared.
    """
    dt = tf / (n_nodes - 1)
    
    # x = [r1, v1, a1, ..., rn, vn, an]
    # Length = n_nodes * 9
    
    def objective(x):
        accs = x.reshape((n_nodes, 9))[:, 6:9]
        return np.sum(np.linalg.norm(accs, axis=1)**2)

    def constraints(x):
        states = x.reshape((n_nodes, 9))
        cons = []
        
        # Dynamics constraints (Trapezoidal)
        for i in range(n_nodes - 1):
            r_i, v_i, a_i = states[i, :3], states[i, 3:6], states[i, 6:9]
            r_next, v_next, a_next = states[i+1, :3], states[i+1, 3:6], states[i+1, 6:9]
            
            g_i = -(mu / np.linalg.norm(r_i)**3) * r_i
            g_next = -(mu / np.linalg.norm(r_next)**3) * r_next
            
            # dot(r) = v
            cons.append(r_next - r_i - 0.5 * dt * (v_i + v_next))
            # dot(v) = g + a
            cons.append(v_next - v_i - 0.5 * dt * (g_i + a_i + g_next + a_next))
            
        # Boundary constraints
        cons.append(states[0, :3] - r0)
        cons.append(states[0, 3:6] - v0)
        cons.append(states[-1, :3] - rf)
        cons.append(states[-1, 3:6] - vf)
        
        return np.concatenate([c.flatten() for c in cons])

    # Initial guess: linear interpolation
    x0 = np.zeros(n_nodes * 9)
    for i in range(n_nodes):
        frac = i / (n_nodes - 1)
        x0[i*9 : i*9+3] = r0 + frac * (rf - r0)
        x0[i*9+3 : i*9+6] = v0 + frac * (vf - v0)

    res = minimize(objective, x0, constraints={'type': 'eq', 'fun': constraints}, method='SLSQP')
    
    if res.success:
        return res.x.reshape((n_nodes, 9))
    else:
        return None
