"""
Debris Avoidance Maneuver Planning.
"""

import numpy as np


def plan_avoidance_maneuver(
    r_sat: np.ndarray,
    v_sat: np.ndarray,
    r_debris: np.ndarray,
    v_debris: np.ndarray,
    safety_radius: float,
    t_encounter: float,
) -> tuple[np.ndarray, float]:
    """
    Plans an impulsive avoidance maneuver (Along-Track).
    Along-Track maneuvers are generally most efficient for phasing to avoid collisions.

    Args:
        r_sat (np.ndarray): Sat position [m]
        v_sat (np.ndarray): Sat velocity [m/s]
        r_debris (np.ndarray): Debris position [m]
        v_debris (np.ndarray): Debris velocity [m/s]
        safety_radius (float): Target miss distance [m]
        t_encounter (float): Time to encounter [s] (Time delta)

    Returns
    -------
        Tuple[np.ndarray, float]: (Delta-V Vector [m/s], Miss distance [m] after maneuver)
    """
    v_mag = np.linalg.norm(v_sat)
    if v_mag < 1e-6:
        raise ValueError("Velocity is too small.")

    # Tangential / Along-Track direction
    t_dir = v_sat / v_mag

    # Current miss distance
    r_rel = r_sat - r_debris
    current_miss = np.linalg.norm(r_rel)

    if current_miss >= safety_radius:
        return np.zeros(3), current_miss

    # Approximate Relative Motion (Clohessy-Wiltshire for Along-Track)
    # Displacement along track is roughly delta_r = 3 * t_encounter * dv_t
    # dv_t = delta_r / (3 * t_encounter)
    # To achieve safety_radius, we want delta_r = safety_radius - current_miss
    delta_r_req = safety_radius - current_miss

    if t_encounter < 10.0:
        dv_mag = (
            delta_r_req / t_encounter
        )  # Impulse at encounter essentially doesn't work, requires high speed
    else:
        # Along-Track phasing approximation
        dv_mag = delta_r_req / (3.0 * t_encounter)

    dv_vec = t_dir * dv_mag

    # Return estimated new miss distance
    estimated_miss = current_miss + np.abs(dv_mag) * (
        3.0 * t_encounter if t_encounter > 10.0 else t_encounter
    )

    return dv_vec, estimated_miss
