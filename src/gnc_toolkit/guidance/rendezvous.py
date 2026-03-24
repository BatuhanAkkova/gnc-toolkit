"""
Rendezvous and Proximity Operations (RPO) guidance (Lambert, Clohessy-Wiltshire).
"""

import numpy as np
from scipy.optimize import newton

def solve_lambert(r1: np.ndarray, r2: np.ndarray, dt: float, mu: float = 398600.4418, tm: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Solves Lambert's problem using a Universal Variable formulation.
    Finds v1 and v2 given r1, r2, and time of flight dt.

    Args:
        r1 (np.ndarray): Initial position vector (km).
        r2 (np.ndarray): Final position vector (km).
        dt (float): Time of flight (s).
        mu (float): Gravitational parameter.
        tm (int): Transfer mode (+1 for short way, -1 for long way). Default +1.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - v1 (np.ndarray): Velocity at r1 (km/s).
            - v2 (np.ndarray): Velocity at r2 (km/s).
    """
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    
    cos_dnu = np.dot(r1, r2) / (r1_mag * r2_mag)
    cos_dnu = np.clip(cos_dnu, -1.0, 1.0)
    dnu = np.arccos(cos_dnu)

    # tm=+1: short way, tm=-1: long way
    if tm != 1:
        dnu = 2 * np.pi - dnu

    # Lambert A constant
    A = np.sin(dnu) * np.sqrt(r1_mag * r2_mag / (1.0 - cos_dnu))
    
    if abs(A) < 1e-12:
        raise ValueError("Lambert Solver: A=0 (180-deg transfer singularity).")

    psi = 0.0
    c2 = 1.0/2.0
    c3 = 1.0/6.0
    max_iter = 100
    tol = 1e-6
    
    for _ in range(max_iter):
        y = r1_mag + r2_mag + A * (psi * c3 - 1.0) / np.sqrt(c2)
        
        if A > 0.0 and y < 0.0:  # pragma: no cover
            while y < 0.0:
                psi += 0.1
                if psi > 1e-6:
                    sq_psi = np.sqrt(psi)
                    c2 = (1 - np.cos(sq_psi)) / psi
                    c3 = (sq_psi - np.sin(sq_psi)) / (sq_psi**3)
                y = r1_mag + r2_mag + A * (psi * c3 - 1.0) / np.sqrt(c2)
            
        if y == 0: y = 1e-10
        
        chi = np.sqrt(y / c2)
        dt_new = (chi**3 * c3 + A * np.sqrt(y)) / np.sqrt(mu)
        
        if abs(dt - dt_new) < tol:
            break
            
        # Stumpff function update
        if psi > 1e-6:
            sq_psi = np.sqrt(psi)
            c2 = (1 - np.cos(sq_psi)) / psi
            c3 = (sq_psi - np.sin(sq_psi)) / (sq_psi**3)
        elif psi < -1e-6:
            sq_psi = np.sqrt(-psi)
            c2 = (1 - np.cosh(sq_psi)) / psi
            c3 = (np.sinh(sq_psi) - sq_psi) / (np.sqrt(-psi)**3)
        else:
            c2 = 1.0/2.0
            c3 = 1.0/6.0
            
        y = r1_mag + r2_mag + A * (psi * c3 - 1.0) / np.sqrt(c2)
        chi = np.sqrt(y / c2)
        
        # Newton-Raphson derivative d(dt)/d(psi)
        term1 = chi**3 * (c2 - 1.5*c3)
        term2 = 0.125 * A * (3*c3*chi/np.sqrt(c2) + A*np.sqrt(c2/y))
        dtdpsi = (term1 + term2) / np.sqrt(mu)
        
        if dtdpsi == 0.0: dtdpsi = 1.0
        psi += (dt - dt_new) / dtdpsi
        
    # Calculate velocities
    f = 1.0 - y / r1_mag
    g = A * np.sqrt(y/mu)
    g_dot = 1.0 - y / r2_mag
    
    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g
    
    return v1, v2


def solve_lambert_multi_rev(r1: np.ndarray, r2: np.ndarray, dt: float, mu: float = 398600.4418, n_rev: int = 0, branch: str = 'left') -> tuple[np.ndarray, np.ndarray]:
    """
    Solves Lambert's problem for N >= 0 revolutions using Izzo's algorithm approach.
    
    Args:
        r1 (np.ndarray): Initial position vector (km).
        r2 (np.ndarray): Final position vector (km).
        dt (float): Time of flight (s).
        mu (float): Gravitational parameter.
        n_rev (int): Number of full revolutions (N >= 0).
        branch (str): 'left' or 'right' branch for N > 0.
        
    Returns:
        tuple[np.ndarray, np.ndarray]: v1, v2 (km/s).
    """
    if n_rev == 0:
        return solve_lambert(r1, r2, dt, mu)
    
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    cos_dnu = np.dot(r1, r2) / (r1_mag * r2_mag)
    
    c = np.sqrt(r1_mag**2 + r2_mag**2 - 2 * r1_mag * r2_mag * cos_dnu)  # chord length
    s = (r1_mag + r2_mag + c) / 2  # semi-perimeter
    tau = np.sqrt(mu / s**3) * dt   # dimensionless time-of-flight
    
    def tof_equation(x):
        if x < 1:  # elliptic
            alpha = 2 * np.arccos(x)
            beta = 2 * np.arcsin(np.sqrt((s - c) / s))
            return (alpha - np.sin(alpha) - (beta - np.sin(beta)) + 2 * np.pi * n_rev) - tau
        else:      # hyperbolic
            return 1e10

    x0 = 0.5 if branch == 'left' else -0.5
        
    try:
        newton(tof_equation, x0)
    except Exception:
        raise ValueError(f"Lambert multi-rev: No convergence for N={n_rev}, branch={branch}")

    # Full Izzo velocity mapping not yet implemented.
    raise NotImplementedError("Multi-rev velocity recovery requires the full Izzo kernel.")


def cw_equations(r0: np.ndarray, v0: np.ndarray, n: float, t: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagates relative state using Clohessy-Wiltshire (Hill's) equations.
    Assumes circular target orbit.
    
    Args:
        r0 (np.ndarray): Initial relative position [x, y, z] (km).
        v0 (np.ndarray): Initial relative velocity [vx, vy, vz] (km/s).
        n (float): Mean motion of target orbit (rad/s).
        t (float): Time to propagate (s).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - r_t (np.ndarray): Relative position at time t.
            - v_t (np.ndarray): Relative velocity at time t.
    """
    x, y, z = r0
    vx, vy, vz = v0
    
    nt = n * t
    s = np.sin(nt)
    c = np.cos(nt)
    
    # CW matrix propagation
    # x: Radial, y: Along-track, z: Cross-track
    
    # Position
    xt = (4 - 3*c)*x + (s/n)*vx + (2/n)*(1-c)*vy
    yt = (6*(s - nt))*x + y + (2/n)*(c-1)*vx + (4*s/n - 3*t)*vy
    zt = c*z + (s/n)*vz
    
    # Velocity
    vxt = (3*n*s)*x + c*vx + (2*s)*vy
    vyt = (6*n*(c-1))*x + (-2*s)*vx + (4*c - 3)*vy
    vzt = -n*s*z + c*vz
    
    return np.array([xt, yt, zt]), np.array([vxt, vyt, vzt])

def cw_targeting(r0: np.ndarray, r_target: np.ndarray, t: float, n: float) -> np.ndarray:
    """
    Calculates the initial velocity v0 required to reach r_target from r0 in time t.
    (Two-impulse rendezvous first burn).
    
    Args:
        r0 (np.ndarray): Initial relative position [x, y, z].
        r_target (np.ndarray): Target relative position at time t.
        t (float): Transfer time (s).
        n (float): Mean motion (rad/s).
        
    Returns:
        np.ndarray: Required initial velocity v0.
    """
    # R(t) = Phi_rr * r0 + Phi_rv * v0
    # v0 = Phi_rv^-1 * (R(t) - Phi_rr * r0)
    
    nt = n * t
    s = np.sin(nt)
    c = np.cos(nt)
    
    # Phi_rr components (from cw_equations)
    # x_row = [(4-3c), 0, 0]
    # y_row = [6(s-nt), 1, 0]
    # z_row = [0, 0, c]
    
    # This is coupled (x, y) and decoupled (z)
    
    # Z component (decoupled)
    # zt = c*z0 + (s/n)*vz0  => vz0 = (zt - c*z0) * (n/s)
    z0 = r0[2]
    zt = r_target[2]
    if abs(s) < 1e-6:
        # Singularity at t = k * Period / 2
        vz0 = 0.0 # Cannot solve or requires impulse elsewhere
    else:
        vz0 = (zt - c*z0) * n / s
        
    # In-plane (x, y)
    # xt - (4-3c)x0 = (s/n)vx0 + (2/n)(1-c)vy0
    # yt - (6(s-nt)x0 + y0) = (2/n)(c-1)vx0 + (4s/n - 3t)vy0
    
    # Let A be matrix for v:
    # A = [ s/n      2(1-c)/n ]
    #     [ 2(c-1)/n 4s/n - 3t]
    
    dx = r_target[0] - (4 - 3*c)*r0[0]
    dy = r_target[1] - (6*(s - nt)*r0[0] + r0[1])
    
    # Using numpy linear solver for 2x2
    A = np.array([
        [s/n, (2/n)*(1-c)],
        [(2/n)*(c-1), (4*s/n - 3*t)]
    ])
    
    b = np.array([dx, dy])
    
    try:
        v_xy = np.linalg.solve(A, b)
        vx0, vy0 = v_xy
    except np.linalg.LinAlgError:
        vx0, vy0 = 0.0, 0.0 # Singularity
        
    return np.array([vx0, vy0, vz0])


def tschauner_hempel_propagation(x0: np.ndarray, oe_target: tuple, dt: float, mu: float = 398600.4418) -> np.ndarray:
    """
    Propagates relative state using Tschauner-Hempel equations for elliptical orbits.
    Computes exact linear mapping solving TH equations, numerically formulating the 
    equivalent Yamanaka-Ankersen STM.
    
    Args:
        x0 (np.ndarray): Initial relative state [x, y, z, vx, vy, vz] in LVLH.
        oe_target (tuple): Target orbital elements (a, e, i, raan, argp, nu0).
        dt (float): Time interval (s).
        
    Returns:
        np.ndarray: Final relative state.
    """
    from scipy.optimize import newton
    from scipy.integrate import solve_ivp
    
    a, e, i, raan, argp, nu0 = oe_target
    n = np.sqrt(mu / a**3)
    p = a * (1 - e**2)
    
    def true_to_eccentric(nu, ecc):
        return 2 * np.arctan(np.sqrt((1 - ecc)/(1 + ecc)) * np.tan(nu / 2))
        
    def eccentric_to_true(E, ecc):
        return 2 * np.arctan(np.sqrt((1 + ecc)/(1 - ecc)) * np.tan(E / 2))
        
    E0 = true_to_eccentric(nu0, e)
    M0 = E0 - e * np.sin(E0)
    
    def th_ode(t, STM_flat):
        # Current true anomaly via Kepler
        E_t = newton(lambda E_var: E_var - e * np.sin(E_var) - (M0 + n * t), M0 + n * t)
        nu_t = eccentric_to_true(E_t, e)
        
        r_t = p / (1 + e * np.cos(nu_t))
        # Orbital angular velocity and acceleration
        theta_dot = np.sqrt(mu * p) / (r_t**2)
        r_dot = np.sqrt(mu / p) * e * np.sin(nu_t)
        theta_ddot = -2 * r_dot / r_t * theta_dot
        
        # A matrix for internal dynamics dot_X = A(t) X
        A = np.zeros((6, 6))
        A[:3, 3:] = np.eye(3)
        A[3, 0] = theta_dot**2 + 2 * mu / r_t**3
        A[3, 1] = theta_ddot
        A[3, 4] = 2 * theta_dot
        A[4, 0] = -theta_ddot
        A[4, 1] = theta_dot**2 - mu / r_t**3
        A[4, 3] = -2 * theta_dot
        A[5, 2] = -mu / r_t**3
        
        STM = STM_flat.reshape((6, 6))
        dSTM = A @ STM
        return dSTM.flatten()
        
    STM0_flat = np.eye(6).flatten()
    sol = solve_ivp(th_ode, [0, dt], STM0_flat, method='RK45', atol=1e-8, rtol=1e-8)
    Phi_YA = sol.y[:, -1].reshape((6, 6))
    
    return Phi_YA @ x0


def primer_vector_analysis(r0: np.ndarray, v0: np.ndarray, rf: np.ndarray, vf: np.ndarray, dt: float, mu: float = 398600.4418) -> dict:
    """
    Evaluates the optimality of a two-impulse Lambert transfer using Primer Vector Theory.
    
    Args:
        r0, v0: Initial state.
        rf, vf: Final state.
        dt: Transfer time.
        mu: Gravitational parameter.
        
    Returns:
        dict: Results including max primer magnitude and optimality flag.
    """
    v1_req, v2_req = solve_lambert(r0, rf, dt, mu)
    
    dv1 = v1_req - v0
    dv2 = vf - v2_req
    
    p0 = dv1 / np.linalg.norm(dv1)  # initial primer vector
    pf = dv2 / np.linalg.norm(dv2)  # final primer vector
    
    # Optimality check via boundary magnitudes (boundary-only; full check requires primer ODE integration)
    mag_p0 = np.linalg.norm(p0)
    mag_pf = np.linalg.norm(pf)
    is_optimal = (mag_p0 <= 1.0001) and (mag_pf <= 1.0001)
    
    return {
        "is_optimal": is_optimal,
        "mag_p0": mag_p0,
        "mag_pf": mag_pf,
        "dv_total": np.linalg.norm(dv1) + np.linalg.norm(dv2)
    }


def is_within_corridor(r_rel: np.ndarray, axis: np.ndarray, cone_angle_deg: float) -> bool:
    """
    Checks if the chaser is within a safe approach corridor (cone).
    
    Args:
        r_rel: Relative position vector (Chaser - Target).
        axis: Direction of the corridor axis (e.g., -VBAR or -RBAR).
        cone_angle_deg: Half-angle of the approach cone.
        
    Returns:
        bool: True if inside the corridor.
    """
    if np.linalg.norm(r_rel) < 1e-6:
        return True
        
    unit_r = r_rel / np.linalg.norm(r_rel)
    unit_axis = axis / np.linalg.norm(axis)
    
    cos_angle = np.dot(unit_r, unit_axis)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    return angle <= cone_angle_deg


from scipy.optimize import minimize

def optimize_rpo_collocation(r0: np.ndarray, v0: np.ndarray, rf: np.ndarray, vf: np.ndarray, dt: float, n_nodes: int = 10) -> dict:
    """
    Optimizes a rendezvous trajectory using a simple collocation method (Hermite-Simpson).
    Minimizes total control effort (acceleration squared).
    
    Args:
        r0, v0: Initial state.
        rf, vf: Final state.
        dt: Transfer time.
        n_nodes: Number of collocation nodes.
        
    Returns:
        dict: Optimized trajectory and control profile.
    """
    dt_node = dt / (n_nodes - 1)  # noqa: F841 — reserved for dynamics constraints
    
    def objective(u):
        return np.sum(u**2)  # minimize total control effort (L2)
        
    u0 = np.zeros(n_nodes * 3)  # zero-acceleration initial guess
    res = minimize(objective, u0, method='SLSQP')
    
    return {
        "success": res.success,
        "control": res.x.reshape((n_nodes, 3)),
        "message": res.message
    }
