import numpy as np
from scipy.integrate import solve_ivp
from gnc_toolkit.disturbances.drag import LumpedDrag
from gnc_toolkit.utils.frame_conversion import eci2geodetic

def calculate_propellant_mass(initial_mass: float, dv: float, isp: float) -> float:
    """
    Calculates the required propellant mass using the Tsiolkovsky rocket equation.
    
    Args:
        initial_mass (float): Initial mass of the spacecraft (kg).
        dv (float): Delta-V required (km/s).
        isp (float): Specific impulse (s).
        
    Returns:
        float: Propellant mass required (kg).
    """
    g0 = 0.00980665 # km/s^2
    if isp <= 0:
        raise ValueError("Isp must be positive.")
    mass_ratio = np.exp(dv / (isp * g0))
    final_mass = initial_mass / mass_ratio
    return initial_mass - final_mass

def calculate_delta_v(initial_mass: float, propellant_mass: float, isp: float) -> float:
    """
    Calculates the Delta-V available from a given propellant mass.
    
    Args:
        initial_mass (float): Initial mass of the spacecraft (kg).
        propellant_mass (float): Mass of the propellant to be consumed (kg).
        isp (float): Specific impulse (s).
        
    Returns:
        float: Delta-V available (km/s).
    """
    g0 = 0.00980665 # km/s^2
    if propellant_mass >= initial_mass:
        raise ValueError("Propellant mass cannot exceed or equal initial mass.")
    if isp <= 0:
        raise ValueError("Isp must be positive.")
    final_mass = initial_mass - propellant_mass
    return isp * g0 * np.log(initial_mass / final_mass)

def calculate_staged_delta_v(stages: list[dict]) -> float:
    """
    Calculates the total Delta-V for a multi-stage rocket.
    
    Args:
        stages (list[dict]): List of dictionaries ordered from first stage (bottom) to last (top).
                             Each dict must contain:
                             - 'm_dry' (float): Dry mass of the stage (kg).
                             - 'm_prop' (float): Propellant mass of the stage (kg).
                             - 'isp' (float): Specific impulse of the stage (s).
                             - 'm0_payload' (float, optional): Payload mass above this stage (kg). If omitted, it's calculated from subsequent stages.
                             
    Returns:
        float: Total Delta-V (km/s).
    """
    # Calculate masses from top to bottom
    total_dv = 0.0
    current_payload = 0.0 # Starts with the final payload
    g0 = 0.00980665 # km/s^2

    # Process stages in reverse order (top stage first)
    reversed_stages = list(reversed(stages))
    for i, stage in enumerate(reversed_stages):
        m_dry = stage['m_dry']
        m_prop = stage['m_prop']
        isp = stage['isp']
        
        # Initial mass of this stage including all stages above it (current_payload)
        m0_stage = m_dry + m_prop + current_payload
        m_f_stage = m_dry + current_payload # final mass after burn
        
        dv_stage = isp * g0 * np.log(m0_stage / m_f_stage)
        total_dv += dv_stage
        
        # update current payload for the next stage down
        current_payload = m0_stage # The entire current stage is the payload for the stage below it
        
    return total_dv

class ManeuverSequence:
    """
    Tracks a sequence of maneuvers and accurately budgets propellant consumption over time.
    """
    def __init__(self, initial_mass: float, isp: float):
        """
        Initialize the ManeuverSequence.
        
        Args:
            initial_mass (float): Initial total mass of the spacecraft (kg).
            isp (float): Specific impulse of the main propulsion system (s).
        """
        self.initial_mass = initial_mass
        self.current_mass = initial_mass
        self.isp = isp
        self.maneuvers = [] # List of tuples (name, dv, m_prop_consumed, m_final, description)

    def add_maneuver(self, name: str, dv: float, description: str = ""):
        """
        Adds a maneuver to the sequence and updates the current mass.
        
        Args:
            name (str): Name of the maneuver.
            dv (float): Delta-V of the maneuver (km/s).
            description (str): Optional description.
        """
        if dv < 0:
            raise ValueError("Delta-V must be non-negative.")
            
        m_prop = calculate_propellant_mass(self.current_mass, dv, self.isp)
        self.current_mass -= m_prop
        
        self.maneuvers.append({
            'name': name,
            'dv': dv,
            'm_prop_consumed': m_prop,
            'm_final': self.current_mass,
            'description': description
        })

    def get_budget_history(self) -> list[dict]:
        """
        Returns the history of the maneuver budget.
        
        Returns:
            list[dict]: List of dictionaries with installment details.
        """
        return self.maneuvers

    def get_total_dv(self) -> float:
        """Total Delta-V applied."""
        return sum(m['dv'] for m in self.maneuvers)

    def get_total_propellant(self) -> float:
        """Total propellant consumed."""
        return self.initial_mass - self.current_mass

def predict_lifetime(r_eci: np.ndarray, v_eci: np.ndarray, mass: float, area: float, cd: float, 
                      density_model, jd_epoch: float, max_days: float = 30, dt: float = 100.0) -> dict:
    """
    Predicts the orbital lifetime due to atmospheric drag using numerical integration.
    Saves and returns the time of reentry (altitude < 100km).
    
    Args:
        r_eci (np.ndarray): Initial position vector [km].
        v_eci (np.ndarray): Initial velocity vector [km/s].
        mass (float): Spacecraft mass [kg].
        area (float): Equivalent cross-sectional area [m^2].
        cd (float): Drag coefficient.
        density_model: Object with `get_density(r_eci, jd)` method.
        jd_epoch (float): Julian Date epoch.
        max_days (float): Maximum prediction duration in days.
        dt (float): Approximate time step for solver output [s].
        
    Returns:
        dict: Breakdown of prediction results:
              - 'reentry_detected' (bool)
              - 'lifetime_days' (float)
              - 'final_altitude' (float)
              - 'trajectory' (dict with 't', 'r', 'v')
    """
    # Initialize LumpedDrag
    # Note: LumpedDrag expects r in meters and v in m/s, acceleration in m/s^2.
    drag_model = LumpedDrag(density_model, co_rotate=True)
    
    # Constants
    RE_EARTH = 6378.137 # km
    MU = 398600.4418 # km^3/s^2
    
    # State: [r_km, v_km_s]
    y0 = np.concatenate([r_eci, v_eci])
    
    def equations_of_motion(t, y):
        # t is in seconds
        r = y[:3] # km
        v = y[3:] # km/s
        
        r_mag = np.linalg.norm(r)
        
        # Two-body gravity
        a_grav = -MU / (r_mag**3) * r # km/s^2
        
        # Drag calculation
        # Convert to SI for LumpedDrag interaction
        r_m = r * 1000.0
        v_m = v * 1000.0
        jd_curr = jd_epoch + t / 86400.0
        
        a_drag_m_s2 = drag_model.get_acceleration(r_m, v_m, jd_curr, mass, area, cd)
        a_drag_km_s2 = a_drag_m_s2 / 1000.0 # Convert back to km/s^2
        
        a_total = a_grav + a_drag_km_s2
        
        return np.concatenate([v, a_total])

    def reentry_event(t, y):
        r = y[:3]
        _, _, h = eci2geodetic(r * 1000.0, jd_epoch + t / 86400.0)
        # h is in meters
        return h - 100000.0 # Event at 100 km altitude
    
    reentry_event.terminal = True
    reentry_event.direction = -1 # altitude decreasing
    
    # Max simulation time in seconds
    t_max = max_days * 86400.0
    
    # Call Scipy solve_ivp
    # Using RK45 for robustness or high drag scenarios
    sol = solve_ivp(
        equations_of_motion,
        (0, t_max),
        y0,
        method='RK45',
        events=reentry_event,
        dense_output=True,
        max_step=dt # Limit step size to avoid missing entry or accurate drag
    )
    
    # Final state processing
    reentry_detected = len(sol.t_events[0]) > 0
    lifetime_s = sol.t[-1]
    final_r = sol.y[:3, -1]
    _, _, final_h = eci2geodetic(final_r * 1000.0, jd_epoch + lifetime_s / 86400.0)
    
    return {
        'reentry_detected': reentry_detected,
        'lifetime_days': lifetime_s / 86400.0,
        'final_altitude': final_h,
        'trajectory': {
            't': sol.t,
            'r': sol.y[:3, :].T,
            'v': sol.y[3:, :].T
        }
    }
