import numpy as np

class LumpedDrag:
    """Lumped drag model."""
    def __init__(self, density_model):
        """
        Args:
            density_model: Object with get_density(r_eci, jd) method.
        """
        self.density_model = density_model

    def get_acceleration(self, r_eci, v_eci, jd, mass, area, cd):
        """
        Calculate drag acceleration.
        
        Args:
            r_eci (np.ndarray): Position ECI [m]
            v_eci (np.ndarray): Velocity ECI [m/s]
            jd (float): Julian Date
            mass (float): Spacecraft mass [kg]
            area (float): Cross-sectional area [m^2]
            cd (float): Drag coefficient
            
        Returns:
            np.ndarray: Acceleration ECI [m/s^2]
        """
        # Calculate density
        rho = self.density_model.get_density(r_eci, jd)
        
        # Velocity relative to rotating atmosphere
        # w_earth = 7.2921159e-5 rad/s
        w_earth = np.array([0, 0, 7.2921159e-5])
        v_rel = v_eci - np.cross(w_earth, r_eci)
        v_rel_norm = np.linalg.norm(v_rel)
        
        # Drag force direction (opposing relative velocity)
        drag_force_mag = 0.5 * rho * (v_rel_norm**2) * cd * area
        drag_acc = -(drag_force_mag / mass) * (v_rel / v_rel_norm)
        
        return drag_acc
