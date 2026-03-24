"""
Initial Orbit Determination (IOD) methods (Gibbs, Gauss, Laplace).
"""

import numpy as np


def gibbs_iod(r1, r2, r3, mu=398600.4415e9):
    """
    Gibbs method for Initial Orbit Determination (3 position vectors).
    Used for long-arc observations (separation > 5 degrees).

    Args:
        r1, r2, r3 (np.ndarray): Three position vectors in ECI [m]
        mu (float): Gravitational parameter

    Returns
    -------
        v2 (np.ndarray): Velocity vector at time t2 [m/s]
    """
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    r3_mag = np.linalg.norm(r3)

    # Check coplanarity
    h_hat = np.cross(r1, r2)
    norm_h = np.linalg.norm(h_hat)
    if norm_h > 1e-12:
        h_hat /= norm_h
    else:
        h_hat = np.zeros(3)

    if abs(np.dot(r3, h_hat)) > 1e-4 * r3_mag:
        pass

    # Gibbs vectors
    D = np.cross(r1, r2) + np.cross(r2, r3) + np.cross(r3, r1)
    N = r1_mag * np.cross(r2, r3) + r2_mag * np.cross(r3, r1) + r3_mag * np.cross(r1, r2)
    L = np.linalg.norm(D)

    if L < 1e-12:
        return np.zeros(3)

    S = r1 * (r2_mag - r3_mag) + r2 * (r3_mag - r1_mag) + r3 * (r1_mag - r2_mag)
    v2 = np.sqrt(mu / (np.linalg.norm(N) * L)) * (np.cross(D, r2) / r2_mag + S)

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

    Returns
    -------
        v2 (np.ndarray): Velocity at t2
    """
    dt31 = dt21 + dt32

    # Avoid Division by Zero
    norm1, norm2, norm3 = np.linalg.norm(r1), np.linalg.norm(r2), np.linalg.norm(r3)
    if norm1 < 1.0 or norm2 < 1.0 or norm3 < 1.0:
        raise ValueError("Position vectors must be non-zero.")

    term1 = -dt32 * (1.0 / (dt21 * dt31) + mu / (12.0 * np.linalg.norm(r1) ** 3)) * r1
    term2 = (dt32 - dt21) * (1.0 / (dt21 * dt32) + mu / (12.0 * np.linalg.norm(r2) ** 3)) * r2
    term3 = dt21 * (1.0 / (dt32 * dt31) + mu / (12.0 * np.linalg.norm(r3) ** 3)) * r3

    v2 = term1 + term2 + term3
    return v2


def gauss_iod(rho_hat1, rho_hat2, rho_hat3, t1, t2, t3, R1, R2, R3, mu=398600.4415e9):
    """
    Gauss method for Initial Orbit Determination (3 line-of-sight vectors).

    Args:
        rho_hat1, rho_hat2, rho_hat3 (np.ndarray): LOS unit vectors
        t1, t2, t3 (float): Observation timestamps [s]
        R1, R2, R3 (np.ndarray): Observer position vectors in ECI [m]
        mu (float): Gravitational parameter

    Returns
    -------
        np.ndarray: [rx, ry, rz, vx, vy, vz] State at t2 [m, m/s]
    """
    # Calculate the time intervals
    tau1 = t1 - t2
    tau3 = t3 - t2
    tau = tau3 - tau1

    # Calculate the cross products
    p1 = np.cross(rho_hat2, rho_hat3)
    p2 = np.cross(rho_hat1, rho_hat3)
    p3 = np.cross(rho_hat1, rho_hat2)

    # Calculate D0
    D0 = np.dot(rho_hat1, p1)
    if abs(D0) < 1e-18:
        raise ValueError("LOS vectors are nearly coplanar (D0 is too small).")

    # Calculate six scalar quantities
    D11 = np.dot(R1, p1)
    D21 = np.dot(R2, p1)
    D31 = np.dot(R3, p1)

    D12 = np.dot(R1, p2)
    D22 = np.dot(R2, p2)
    D32 = np.dot(R3, p2)

    D13 = np.dot(R1, p3)
    D23 = np.dot(R2, p3)
    D33 = np.dot(R3, p3)

    # Calculate A and B
    A = (1.0 / D0) * (-(tau3 / tau) * D12 + D22 + (tau1 / tau) * D32)
    B = (1.0 / (6.0 * D0)) * (
        (tau3**2 - tau**2) * (tau3 / tau) * D12 + (tau**2 - tau1**2) * (tau1 / tau) * D32
    )

    # Calculate E and R2^2
    E = np.dot(R2, rho_hat2)

    # Calculate a, b and c
    a_coeff = -(A**2 + 2.0 * A * E + np.dot(R2, R2))
    b_coeff = -2.0 * mu * B * (A + E)
    c_coeff = -(mu**2) * B**2

    # Find roots: x^8 + ax^6 + bx^3 + c = 0
    poly_coeffs = [1.0, 0.0, a_coeff, 0.0, 0.0, b_coeff, 0.0, 0.0, c_coeff]
    roots = np.roots(poly_coeffs)
    real_positive_roots = roots[np.isreal(roots) & (roots.real > 0)].real
    if len(real_positive_roots) == 0:
        raise ValueError("No physical (positive real) root found for Gauss radius r2.")
    r2 = np.max(real_positive_roots)
    if r2 < 3000e3 or r2 > 100000e3:
        raise ValueError("Radius out of bounds")

    # Calculate initial rho1, rho2, rho3
    u = mu / r2**3
    f1 = 1.0 - 0.5 * u * tau1**2
    f3 = 1.0 - 0.5 * u * tau3**2
    g1 = tau1 - (1.0 / 6.0) * u * tau1**3
    g3 = tau3 - (1.0 / 6.0) * u * tau3**3

    denom = f1 * g3 - f3 * g1
    c1, c3 = g3 / denom, -g1 / denom

    mat = np.array([c1 * rho_hat1, -rho_hat2, c3 * rho_hat3]).T
    rhs = R2 - c1 * R1 - c3 * R3
    rho1, rho2, rho3 = np.linalg.solve(mat, rhs)

    # Initial position vectors
    r1_vec = R1 + rho1 * rho_hat1
    r2_vec = R2 + rho2 * rho_hat2
    r3_vec = R3 + rho3 * rho_hat3

    # Initial velocity vector
    v2_vec = (1.0 / denom) * (-f3 * r1_vec + f1 * r3_vec)

    # Iterative Refinement
    rho1_old, rho2_old, rho3_old = rho1, rho2, rho3
    n, nmax, tol = 0, 1000, 1e-8
    diff1 = diff2 = diff3 = 1.0
    alpha_relax = 0.1

    r2_vec_best = r2_vec.copy()
    v2_vec_best = v2_vec.copy()
    min_diff = 1e20

    while (diff1 > tol or diff2 > tol or diff3 > tol) and (n < nmax):
        n += 1
        ro = np.linalg.norm(r2_vec)

        # Guard against non-physical divergence
        if ro < 5000e3 or ro > 100000e3:
            break

        vo = np.linalg.norm(v2_vec)
        vro = np.dot(v2_vec, r2_vec) / ro
        alpha = 2.0 / ro - vo**2 / mu

        # Solve universal Kepler's equation
        x1 = _kepler_U(tau1, ro, vro, alpha, mu)
        x3 = _kepler_U(tau3, ro, vro, alpha, mu)

        f1, g1 = _get_fg(x1, tau1, ro, alpha, mu)
        f3, g3 = _get_fg(x3, tau3, ro, alpha, mu)

        denom = f1 * g3 - f3 * g1
        c1, c3 = g3 / denom, -g1 / denom

        mat = np.array([c1 * rho_hat1, -rho_hat2, c3 * rho_hat3]).T
        rhs = R2 - c1 * R1 - c3 * R3
        try:
            sol = np.linalg.solve(mat, rhs)
            rho1_new, rho2_new, rho3_new = sol[0], sol[1], sol[2]

            # Apply tight relaxation damping
            rho1 = alpha_relax * rho1_new + (1 - alpha_relax) * rho1
            rho2 = alpha_relax * rho2_new + (1 - alpha_relax) * rho2
            rho3 = alpha_relax * rho3_new + (1 - alpha_relax) * rho3

            r1_vec = R1 + rho1 * rho_hat1
            r2_vec = R2 + rho2 * rho_hat2
            r3_vec = R3 + rho3 * rho_hat3

            v2_new = (1.0 / denom) * (-f3 * r1_vec + f1 * r3_vec)
            v2_vec = alpha_relax * v2_new + (1 - alpha_relax) * v2_vec

            diff1, diff2, diff3 = abs(rho1 - rho1_old), abs(rho2 - rho2_old), abs(rho3 - rho3_old)
            rho1_old, rho2_old, rho3_old = rho1, rho2, rho3
        except np.linalg.LinAlgError:
            break

    # Final Hybrid Velocity: Herrick-Gibbs on converged positions for short arcs
    r1_vec = R1 + rho1 * rho_hat1
    r2_vec = R2 + rho2 * rho_hat2
    r3_vec = R3 + rho3 * rho_hat3
    try:
        v2_vec = herrick_gibbs_iod(r1_vec, r2_vec, r3_vec, -tau1, tau3, mu)
    except:
        v2_vec = (1.0 / denom) * (-f3 * r1_vec + f1 * r3_vec)

    return np.concatenate([r2_vec, v2_vec])


def _stumpff(psi):
    """Calculate Stumpff functions C2 and C3 with series fallback for small psi."""
    if psi > 1e-6:
        sqrt_psi = np.sqrt(psi)
        c2 = (1.0 - np.cos(sqrt_psi)) / psi
        c3 = (sqrt_psi - np.sin(sqrt_psi)) / (psi * sqrt_psi)
    elif psi < -1e-6:
        sqrt_psi = np.sqrt(-psi)
        c2 = (1.0 - np.cosh(sqrt_psi)) / psi
        c3 = (np.sinh(sqrt_psi) - sqrt_psi) / ((-psi) * sqrt_psi)
    else:
        # Maclaurin series expansion
        c2 = 0.5 - psi / 24.0 + psi**2 / 720.0
        c3 = 1.0 / 6.0 - psi / 120.0 + psi**2 / 5040.0
    return c2, c3


def _get_fg(x, tau, r0, alpha, mu):
    """Calculate exact f and g from universal anomaly x."""
    psi = x**2 * alpha
    c2, c3 = _stumpff(psi)
    f = 1.0 - (x**2 / r0) * c2
    g = tau - (x**3 / np.sqrt(mu)) * c3
    return f, g


def _kepler_U(tau, r0, vr0, alpha, mu, tol=1e-8, max_iter=100):
    """Solve universal Kepler's equation for x using Newton-Raphson."""
    x = np.sqrt(mu) * np.abs(alpha) * tau if abs(alpha) > 1e-9 else np.sqrt(mu) * tau / r0

    for _ in range(max_iter):
        psi = x**2 * alpha
        c2, c3 = _stumpff(psi)

        r_dot_v_sqrt_mu = r0 * vr0 / np.sqrt(mu)
        val = (
            r_dot_v_sqrt_mu * x**2 * c2
            + (1.0 - alpha * r0) * x**3 * c3
            + r0 * x
            - np.sqrt(mu) * tau
        )
        deriv = r_dot_v_sqrt_mu * x * (1.0 - psi * c3) + (1.0 - alpha * r0) * x**2 * c2 + r0

        if abs(deriv) < 1e-12:  # Avoid ZeroDivision
            break  # pragma: no cover

        dx = val / deriv
        x -= dx
        if abs(dx) < tol:
            break
    return x


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

    Returns
    -------
        np.ndarray: [rx, ry, rz, vx, vy, vz] State vector in ECI [m, m/s]
    """
    # Determinants for Laplace method
    D = np.linalg.det(np.array([rho_hat, rho_hat_dot, rho_hat_ddot]))

    if abs(D) < 1e-18:
        raise ValueError(
            "Determinant D is too small; observations may be nearly coplanar or insufficient."
        )

    D1 = np.linalg.det(np.array([rho_hat, rho_hat_dot, R_ddot]))
    D2 = np.linalg.det(np.array([rho_hat, rho_hat_dot, R]))

    A = -D1 / D
    B = -mu * D2 / D

    R_mag = np.linalg.norm(R)
    cos_phi = np.dot(rho_hat, R) / R_mag

    # Solve 8th order polynomial for r (radius of satellite)
    poly = [
        1.0,  # r^8
        0.0,  # r^7
        -(A**2 + 2.0 * R_mag * A * cos_phi + R_mag**2),  # r^6
        0.0,  # r^5
        0.0,  # r^4
        -(2.0 * A * B + 2.0 * R_mag * B * cos_phi),  # r^3
        0.0,  # r^2
        0.0,  # r^1
        -(B**2),  # r^0
    ]

    roots = np.roots(poly)
    real_positive_roots = roots[np.isreal(roots) & (roots.real > 0)].real
    if len(real_positive_roots) == 0:
        raise ValueError("No physical (positive real) root found for Laplace IOD.")

    r_mag = real_positive_roots[0]

    rho = A + B / r_mag**3

    # Position
    r_vec = rho * rho_hat + R

    # Velocity approximation using Laplace derivatives
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
    l1_dot = -tau32 / (-tau21 * -tau31)
    l2_dot = (tau21 - tau32) / (tau21 * -tau32)
    l3_dot = tau21 / (tau31 * tau32)

    rho_hat_dot = l1_dot * L1 + l2_dot * L2 + l3_dot * L3
    R_dot = l1_dot * R1 + l2_dot * R2 + l3_dot * R3

    # Second derivatives at t2
    l1_ddot = 2.0 / (-tau21 * -tau31)
    l2_ddot = 2.0 / (tau21 * -tau32)
    l3_ddot = 2.0 / (tau31 * tau32)

    rho_hat_ddot = l1_ddot * L1 + l2_ddot * L2 + l3_ddot * L3

    # Observer acceleration estimation (or use gravity if R is ECI)
    R_ddot = l1_ddot * R1 + l2_ddot * R2 + l3_ddot * R3

    return laplace_iod(L2, rho_hat_dot, rho_hat_ddot, R2, R_dot, R_ddot, mu)
