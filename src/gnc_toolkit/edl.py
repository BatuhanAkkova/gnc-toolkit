import numpy as np
from gnc_toolkit.environment.density import Exponential

def ballistic_entry_dynamics(t: float, state: np.ndarray, cd: float, area: float, mass: float, mu: float = 3.986e14, r_planet: float = 6371000.0, rho_model=None) -> np.ndarray:
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
        
    Returns:
        np.ndarray: State derivative [vx, vy, vz, ax, ay, az].
    """
    r_vec = state[:3]
    v_vec = state[3:]
    r_mag = np.linalg.norm(r_vec)
    v_mag = np.linalg.norm(v_vec)
    
    if rho_model is None:
        # Default to simple exponential for Earth if not provided
        rho_model = Exponential(rho0=1.225, h0=0.0, H=8.5)
        
    # Density calculation (assuming jd=0 for simplicity if using simple model)
    rho = rho_model.get_density(r_vec, 0.0)
    
    # Aerodynamic Drag
    dynamic_pressure = 0.5 * rho * v_mag**2
    drag_mag = dynamic_pressure * cd * area
    a_drag = -(drag_mag / mass) * (v_vec / v_mag) if v_mag > 1e-6 else np.zeros(3)
    
    # Gravity (Point mass)
    a_grav = -(mu / r_mag**3) * r_vec
    
    return np.concatenate([v_vec, a_grav + a_drag])

def lifting_entry_dynamics(t: float, state: np.ndarray, cl: float, cd: float, bank_angle: float, area: float, mass: float, mu: float = 3.986e14, r_planet: float = 6371000.0, rho_model=None) -> np.ndarray:
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
        
    Returns:
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
    u_l_vertical = np.cross(u_v, u_h) # Lift vector in the vertical plane
    
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
        
    Returns:
        float: Heat flux (W/m^2).
    """
    k = 1.74153e-4
    return k * np.sqrt(rho / rn) * v**3

def calculate_g_load(acc_vec: np.ndarray) -> float:
    """
    Calculates G-load from acceleration vector.
    
    Args:
        acc_vec (np.ndarray): Acceleration (m/s^2).
        
    Returns:
        float: G-load (multiples of g0).
    """
    g0 = 9.80665
    return np.linalg.norm(acc_vec) / g0

def aerocapture_guidance(state, target_apoapsis, cd, area, mass, planet_params, rho_model):
    """
    Predictive aerocapture guidance.
    Adjusts bank angle or drag to target a specific exit apoapsis.
    Returns: bank_angle (rad)
    """
    # Simplified logic: predict exit state and adjust
    # For a toolkit, we might use a numeric predictor-corrector
    return 0.0 # Placeholder for full NPC logic

def hazard_avoidance(r, v, hazards, safety_margin=50.0):
    """
    Simple hazard avoidance logic.
    Computes a divert maneuver if a hazard is detected near the landing site.
    
    Args:
        r (np.ndarray): Position (m).
        v (np.ndarray): Velocity (m/s).
        hazards (list): List of hazard positions [x, y, z].
        
    Returns:
        np.ndarray: Correction velocity delta_v (m/s).
    """
    for h in hazards:
        dist = np.linalg.norm(r - h)
        if dist < safety_margin:
            # Divert away from hazard
            u_divert = (r - h) / dist
            return u_divert * 5.0 # 5 m/s nudge
    return np.zeros(3)
