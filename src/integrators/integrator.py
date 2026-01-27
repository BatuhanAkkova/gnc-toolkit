from abc import ABC, abstractmethod
import numpy as np

class Integrator(ABC):
    """
    Abstract base class for numerical integrators.
    """
    @abstractmethod
    def step(self, f, t, y, dt, **kwargs):
        """
        Perform a single integration step.
        
        Args:
            f: Function callable f(t, y, **kwargs) -> dy/dt
            t: Current time
            y: Current state vector
            dt: Time step info (can be a float for fixed step or tuple for adaptive)
            **kwargs: Additional arguments to pass to f
            
        Returns:
            y_next: State at t + dt
            t_next: Time at t + dt
            dt_next: Suggested next time step (for variable step methods), or dt used
        """
        pass

    def integrate(self, f, t_span, y0, dt=0.01, **kwargs):
        """
        Integrate over a time span.
        
        Args:
            f: Function callable f(t, y, **kwargs) -> dy/dt
            t_span: Tuple (t0, tf)
            y0: Initial state vector
            dt: Initial time step
            **kwargs: Additional arguments to pass to f
            
        Returns:
            t_values: Array of time points
            y_values: Array of state vectors
        """
        t0, tf = t_span
        t = t0
        y = np.array(y0)
        
        t_values = [t]
        y_values = [y]
        
        while t < tf:
            # Adjust last step to land exactly on tf
            if t + dt > tf:
                dt = tf - t
            
            y, t, dt = self.step(f, t, y, dt, **kwargs)
            
            t_values.append(t)
            y_values.append(y)
            
        return np.array(t_values), np.array(y_values)
