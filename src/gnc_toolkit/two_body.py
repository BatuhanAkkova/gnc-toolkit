import numpy as np

def two_body_kepler(r_i, v_i, dt, n_iter=100):
    """
    Solves the two-body problem using the Kepler equation.
    Vallado implementation.
    """
    mu = 398600.4418e9 # Earth's gravitational parameter in m^3/s^2
    r_i_mag = np.linalg.norm(r_i)
    v_i_mag = np.linalg.norm(v_i)
    energy = 0.5 * v_i_mag**2 - mu/r_i_mag # Specific orbital energy

    alpha = -2 * energy / mu # Reciprocal of semi-major axis
    a = -mu / (2*energy) if np.abs(energy) > 1e-8 else np.inf
    alpha = 0 if np.abs(alpha) < 1e-8 else alpha

    if alpha > 1e-8: # Circular or Elliptical
        T = 2 * np.pi * np.sqrt(np.abs(alpha)**3 / mu) # Period
        
        if np.abs(dt) > np.abs(T):
            dt = dt % T
        
        x_i = np.sqrt(mu) * dt * alpha

    elif np.abs(alpha) < 1e-8: # Parabolic
        h = np.cross(r_i, v_i)
        h_mag = np.linalg.norm(h)
        p = h_mag**2 / mu
        
        s = 0.5 * (np.pi / 2 - np.arctan(3 * np.sqrt(mu / (p**3)) * dt))
        w = np.arctan(np.tan(s) ** (1/3))
        
        x_i = np.sqrt(p) * (2 / np.tan(2*w))
        alpha = 0

    else: # Hyperbolic
        temp = (-2 * mu * alpha * dt / (np.dot(r_i, v_i) + np.sign(dt) * np.sqrt(-mu/alpha) * (1 - r_i_mag * alpha)))
        x_i = np.sign(dt) * np.sqrt(-1/alpha) * np.log(temp)

    # Newton-Raphson iteration
    i = 0
    while i < n_iter:
        yaw = (x_i**2) * alpha

        c2, c3 = c23_eq(yaw)
        
        nom = np.sqrt(mu) * dt - x_i**3 * c3 - np.dot(r_i, v_i) / np.sqrt(mu) * x_i**2 * c2 - r_i_mag * x_i * (1 - yaw*c3)
        denom = x_i**2 * c2 + np.dot(r_i, v_i) / np.sqrt(mu) * x_i * (1 - yaw*c3) + r_i_mag * (1 - yaw*c2)
        x_new = x_i + nom/denom
        
        if np.abs(x_new - x_i) < 1e-8:
            break
        x_i = x_new
        i += 1

    f = 1 - x_new**2 / r_i_mag * c2
    g = dt - x_new**3 / np.sqrt(mu) * c3
    r_new = f * r_i + g * v_i
    r_new_mag = np.linalg.norm(r_new)
    
    fdot = np.sqrt(mu) / (r_new_mag * r_i_mag) * x_new * (yaw*c3 - 1)
    gdot = 1 - (x_new**2/r_new_mag) * c2
    v_new = fdot * r_i + gdot * v_i

    return r_new, v_new

def c23_eq(yaw):
    """Calculates c2 and c3 functions."""
    if yaw > 1e-8:
        c2 = (1 - np.cos(np.sqrt(yaw))) / yaw
        c3 = (np.sqrt(yaw) - np.sin(np.sqrt(yaw))) / (yaw**(1.5))
    elif yaw < -1e-8:
        c2 = (1 - np.cosh(np.sqrt(-yaw))) / yaw
        c3 = (np.sinh(np.sqrt(-yaw)) - np.sqrt(-yaw)) / ((-yaw)**(1.5))
    else:
        c2 = 0.5
        c3 = 1/6
    return c2, c3
