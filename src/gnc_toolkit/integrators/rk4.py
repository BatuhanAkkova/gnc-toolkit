import numpy as np
from .integrator import Integrator

class RK4(Integrator):
    """
    Runge-Kutta 4th Order Integrator (Fixed Step).
    """

    def step(self, f, t, y, dt, **kwargs):
        """
        Perform a single RK4 step.
        """
        k1 = f(t, y, **kwargs)
        k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1, **kwargs)
        k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2, **kwargs)
        k4 = f(t + dt, y + dt * k3, **kwargs)
        
        y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t_next = t + dt
        
        return y_next, t_next, dt
