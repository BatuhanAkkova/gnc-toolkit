"""
Entry, Descent, and Landing (EDL) dynamics and utilities.
"""

import numpy as np

from gnc_toolkit.environment.density import Exponential


def ballistic_entry_dynamics(
    t: float,
    state: np.ndarray,
    cd: float,
    area: float,
    mass: float,
    mu: float = 3.986e14,
    r_planet: float = 6371000.0,
    rho_model=None,
) -> np.ndarray:
    """
    ODE for ballistic entry dynamics in Cartesian coordinates.

    Args:
        t (float): Time (s).
        state (np.ndarray): [x, y, z, vx, vy, vz] (m, m/s).
        cd (float): Drag coefficient.
        area (float): Reference area (m^2).
        mass (float): Mass (kg).
        mu (float): Gravitational parameter (m^3/s^2).
        r_planet (float): Planet radius (m).
        rho_model: Object with get_density(r_eci, jd) method.

    Returns
    -------
        np.ndarray: State derivative [vx, vy, vz, ax, ay, az].
    """
    r_vec = state[:3]
    v_vec = state[3:]
    r_mag = np.linalg.norm(r_vec)
    v_mag = np.linalg.norm(v_vec)

    if rho_model is None:
        # Default to simple exponential for Earth if not provided
        rho_model = Exponential(rho0=1.225, h0=0.0, H=8.5)

    rho = rho_model.get_density(r_vec, 0.0)  # jd=0 placeholder for simple models

    # Aerodynamic Drag
    dynamic_pressure = 0.5 * rho * v_mag**2
    drag_mag = dynamic_pressure * cd * area
    a_drag = -(drag_mag / mass) * (v_vec / v_mag) if v_mag > 1e-6 else np.zeros(3)

    # Gravity (Point mass)
    a_grav = -(mu / r_mag**3) * r_vec

    return np.concatenate([v_vec, a_grav + a_drag])


def lifting_entry_dynamics(
    t: float,
    state: np.ndarray,
    cl: float,
    cd: float,
    bank_angle: float,
    area: float,
    mass: float,
    mu: float = 3.986e14,
    r_planet: float = 6371000.0,
    rho_model=None,
) -> np.ndarray:
    """
    ODE for lifting entry dynamics in Cartesian coordinates.

    Args:
        t (float): Time (s).
        state (np.ndarray): [x, y, z, vx, vy, vz] (m, m/s).
        cl, cd: Lift and Drag coefficients.
        bank_angle (float): Bank angle (rad).
        area (float): Reference area (m^2).
        mass (float): Mass (kg).
        mu, r_planet: Planet parameters.
        rho_model: Density model.

    Returns
    -------
        np.ndarray: State derivative.
    """
    r_vec = state[:3]
    v_vec = state[3:]
    r_mag = np.linalg.norm(r_vec)
    v_mag = np.linalg.norm(v_vec)

    if rho_model is None:
        rho_model = Exponential(rho0=1.225, h0=0.0, H=8.5)

    rho = rho_model.get_density(r_vec, 0.0)
    dynamic_pressure = 0.5 * rho * v_mag**2

    # Unit vectors for Drag, Lift
    u_v = v_vec / v_mag if v_mag > 1e-6 else np.zeros(3)
    u_h = np.cross(r_vec, v_vec)
    u_h = u_h / np.linalg.norm(u_h) if np.linalg.norm(u_h) > 1e-6 else np.zeros(3)
    u_l_vertical = np.cross(u_v, u_h)  # Lift vector in the vertical plane

    # Rotate lift vector by bank angle
    # L = L_mag * (cos(bank) * u_l_vertical + sin(bank) * u_h)
    lift_mag = dynamic_pressure * cl * area
    drag_mag = dynamic_pressure * cd * area

    a_drag = -(drag_mag / mass) * u_v
    a_lift = (lift_mag / mass) * (np.cos(bank_angle) * u_l_vertical + np.sin(bank_angle) * u_h)

    # Gravity
    a_grav = -(mu / r_mag**3) * r_vec

    return np.concatenate([v_vec, a_grav + a_drag + a_lift])


def sutton_grave_heating(rho: float, v: float, rn: float) -> float:
    """
    Stagnation point heat flux estimation (W/m^2) using Sutton-Grave formula.
    Simplified constant for Earth: k = 1.74153e-4 (for cooling Rn in meters)
    q = k * sqrt(rho/rn) * v^3

    Args:
        rho (float): Density (kg/m^3).
        v (float): Velocity (m/s).
        rn (float): Nose radius (m).

    Returns
    -------
        float: Heat flux (W/m^2).
    """
    k = 1.74153e-4
    return k * np.sqrt(rho / rn) * v**3


def calculate_g_load(acc_vec: np.ndarray) -> float:
    """
    Calculates G-load from acceleration vector.

    Args:
        acc_vec (np.ndarray): Acceleration (m/s^2).

    Returns
    -------
        float: G-load (multiples of g0).
    """
    g0 = 9.80665
    return np.linalg.norm(acc_vec) / g0


def aerocapture_guidance(state, target_apoapsis, cd, area, mass, planet_params, rho_model, cl=0.0):
    """
    Predictive aerocapture guidance.
    Adjusts bank angle (if cl > 0) to target a specific exit apoapsis using numerical predictor-corrector.
    Returns: bank_angle (rad)
    """
    mu = planet_params.get("mu", 3.986e14)
    r_planet = planet_params.get("r_planet", 6371000.0)
    atm_interface = r_planet + 120000.0  # Assumed 120km interface

    from scipy.integrate import solve_ivp

    def get_apoapsis_from_state(s):
        r_vec, v_vec = s[:3], s[3:]
        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)
        energy = v**2 / 2 - mu / r
        if energy >= 0:
            return np.inf  # Escape trajectory
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)
        a = -mu / (2 * energy)
        e = np.sqrt(max(0.0, 1 - h**2 / (a * mu)))
        return a * (1 + e) - r_planet

    if cl <= 0.0:
        return 0.0  # Cannot modulate bank angle without lift

    def predict_apoapsis(bank_angle):
        def dynamics(t, s):
            return lifting_entry_dynamics(
                t, s, cl, cd, bank_angle, area, mass, mu, r_planet, rho_model
            )

        def exit_event(t, s):
            return np.linalg.norm(s[:3]) - atm_interface

        exit_event.terminal = True
        exit_event.direction = 1

        # Integrate forward to exit
        sol = solve_ivp(
            dynamics, (0, 2000.0), state, events=[exit_event], max_step=10.0, rtol=1e-3, atol=1e-3
        )
        return get_apoapsis_from_state(sol.y[:, -1])

    bank_up = 0.0
    bank_down = np.pi
    best_bank = 0.0

    # Simple bisection search on bank angle
    for _ in range(10):
        bank_mid = (bank_up + bank_down) / 2
        ap_mid = predict_apoapsis(bank_mid)
        best_bank = bank_mid

        if np.isinf(ap_mid) or ap_mid > target_apoapsis:
            bank_up = bank_mid  # Need more pull-down (higher atmospheric time)
        else:
            bank_down = bank_mid

        if not np.isinf(ap_mid) and abs(ap_mid - target_apoapsis) < 1000.0:
            break

    return best_bank


def hazard_avoidance(r, v, hazards, safety_margin=50.0):
    """
    Simple hazard avoidance logic.
    Computes a divert maneuver if a hazard is detected near the landing site.

    Args:
        r (np.ndarray): Position (m).
        v (np.ndarray): Velocity (m/s).
        hazards (list): List of hazard positions [x, y, z].

    Returns
    -------
        np.ndarray: Correction velocity delta_v (m/s).
    """
    for h in hazards:
        dist = np.linalg.norm(r - h)
        if dist < safety_margin:
            # Divert away from hazard
            u_divert = (r - h) / dist
            return u_divert * 5.0  # 5 m/s nudge
    return np.zeros(3)
