import numpy as np

def hohmann_transfer(r1: float, r2: float, mu: float = 398600.4418) -> tuple[float, float, float]:
    """
    Calculates the Delta-V requirements and transfer time for a Hohmann transfer
    between two circular orbits.

    Args:
        r1 (float): Radius of the initial orbit (km).
        r2 (float): Radius of the final orbit (km).
        mu (float): Gravitational parameter (km^3/s^2). Default is Earth.

    Returns:
        tuple[float, float, float]:
            - dv1 (float): Delta-V for the first burn (km/s).
            - dv2 (float): Delta-V for the second burn (km/s).
            - transfer_time (float): Time of flight for the transfer (s).
    """
    # Semi-major axis of the transfer ellipse
    a_trans = (r1 + r2) / 2.0
    
    # Velocity at periapsis and apoapsis of transfer orbit
    v_trans_p = np.sqrt(mu * (2/r1 - 1/a_trans))
    v_trans_a = np.sqrt(mu * (2/r2 - 1/a_trans))
    
    # Velocity of initial and final circular orbits
    v_c1 = np.sqrt(mu / r1)
    v_c2 = np.sqrt(mu / r2)
    
    # Calculate Delta-Vs
    dv1 = abs(v_trans_p - v_c1)
    dv2 = abs(v_c2 - v_trans_a)
    
    # Transfer time (half period)
    transfer_time = np.pi * np.sqrt(a_trans**3 / mu)
    
    return dv1, dv2, transfer_time

def bi_elliptic_transfer(r1: float, r2: float, rb: float, mu: float = 398600.4418) -> tuple[float, float, float, float]:
    """
    Calculates the Delta-V requirements and transfer time for a Bi-Elliptic transfer.
    
    Args:
        r1 (float): Radius of the initial orbit (km).
        r2 (float): Radius of the final orbit (km).
        rb (float): Radius of the intermediate apogee (km). Must be > max(r1, r2).
        mu (float): Gravitational parameter (km^3/s^2). Default is Earth.
        
    Returns:
        tuple[float, float, float, float]:
            - dv1 (float): Delta-V for first burn (km/s).
            - dv2 (float): Delta-V for second burn (km/s).
            - dv3 (float): Delta-V for third burn (km/s).
            - transfer_time (float): Total time of flight (s).
    """
    # First transfer: r1 to rb
    a1 = (r1 + rb) / 2.0
    v_c1 = np.sqrt(mu / r1)
    v_trans1_p = np.sqrt(mu * (2/r1 - 1/a1))
    dv1 = abs(v_trans1_p - v_c1)
    
    v_trans1_a = np.sqrt(mu * (2/rb - 1/a1))
    
    # Second transfer: rb to r2
    a2 = (r2 + rb) / 2.0
    v_trans2_a = np.sqrt(mu * (2/rb - 1/a2))
    dv2 = abs(v_trans2_a - v_trans1_a) # Burn at apoapsis rb
    
    v_trans2_p = np.sqrt(mu * (2/r2 - 1/a2))
    v_c2 = np.sqrt(mu / r2)
    dv3 = abs(v_c2 - v_trans2_p)
    
    # Total time
    t1 = np.pi * np.sqrt(a1**3 / mu)
    t2 = np.pi * np.sqrt(a2**3 / mu)
    transfer_time = t1 + t2
    
    return dv1, dv2, dv3, transfer_time

def phasing_maneuver(a: float, T_phase: float, mu: float = 398600.4418) -> tuple[float, float]:
    """
    Calculates the Delta-V required for a phasing maneuver to correct a timing error
    or shift position in the orbit.
    
    Args:
        a (float): Semi-major axis of the current circular orbit (km).
        T_phase (float): Desired time difference to generate (s). 
                         Positive to wait (increase period), negative to catch up.
                         Usually T_phase is small relative to orbital period.
        mu (float): Gravitational parameter.
        
    Returns:
        tuple[float, float]:
            - total_dv (float): Total Delta-V (entry + exit) (km/s).
            - t_wait (float): Duration of the phasing orbit (s).
    """
    # Current period
    T_curr = 2 * np.pi * np.sqrt(a**3 / mu)
    
    # Target period for the phasing orbit
    # If intended to move "forward" (catch up), need a shorter period (faster).
    # If intended to move "backward" (wait), need a longer period (slower).
    T_phasing = T_curr + T_phase
    
    # Semi-major axis of phasing orbit
    a_phasing = (mu * (T_phasing / (2 * np.pi))**2)**(1/3)
    
    # Velocity in current circular orbit
    v_c = np.sqrt(mu / a)
    
    # Velocity at periapsis (or apoapsis) of phasing orbit at the connection point
    # Energy conservation: v^2/2 - mu/r = -mu/2a
    # burn at radius 'a'.
    v_phasing = np.sqrt(mu * (2/a - 1/a_phasing))
    
    dv_burn = abs(v_phasing - v_c)
    
    # perform this burn to enter phasing orbit, and same burn to exit
    total_dv = 2 * dv_burn
    
    return total_dv, T_phasing

def plane_change(v: float, delta_i: float) -> float:
    """
    Calculates Delta-V for a simple plane change maneuver.
    
    Args:
        v (float): Current velocity magnitude (km/s).
        delta_i (float): Plane change angle (radians).
        
    Returns:
        float: Delta-V required (km/s).
    """
    return 2 * v * np.sin(delta_i / 2.0)

def combined_plane_change(v1: float, v2: float, delta_i: float) -> float:
    """
    Calculates Delta-V for a combined maneuver (changing velocity magnitude and inclination).
    Using Law of Cosines.
    
    Args:
        v1 (float): Initial velocity magnitude (km/s).
        v2 (float): Final velocity magnitude (km/s).
        delta_i (float): Plane change angle (radians).
        
    Returns:
        float: Delta-V required (km/s).
    """
    return np.sqrt(v1**2 + v2**2 - 2 * v1 * v2 * np.cos(delta_i))
