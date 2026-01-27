import numpy as np
from .integrator import Integrator
from . import dop853_coeffs as coeffs

class RK853(Integrator):
    """
    Dormand-Prince 8(5,3) Variable Step Integrator.
    High order integrator for high precision requirements.
    """
    
    def __init__(self, rtol=1e-9, atol=1e-12, safety_factor=0.9, min_factor=0.2, max_factor=10.0):
        self.rtol = rtol
        self.atol = atol
        self.safety_factor = safety_factor
        self.min_factor = min_factor
        self.max_factor = max_factor
        
        self.n_stages = coeffs.N_STAGES
        self.A = coeffs.A
        self.B = coeffs.B
        self.C = coeffs.C
        self.E3 = coeffs.E3
        self.E5 = coeffs.E5
        
        # Scipy uses K (stages + 1)
        self.error_exponent = -1 / (7 + 1) # error estimator order is 7

    def step(self, f, t, y, dt, **kwargs):
        """
        Perform a single adaptive RK853 step.
        """
        dt_current = dt
        n = len(y)
        
        # K table: stages + 1 rows, n columns
        # Allocated once to avoid reallocation during step rejection
        K = np.zeros((self.n_stages + 1, n))

        while True:
            # First stage K[0] is f(t, y)
            K[0] = f(t, y, **kwargs)
            
            for s in range(1, self.n_stages + 1):
                # Calculate dy for stage s
                dy = np.dot(coeffs.A[s, :s], K[:s]) * dt_current
                
                K[s] = f(t + coeffs.C[s] * dt_current, y + dy, **kwargs)
                
            # Compute y_new (8th order)
            y_next = y + dt_current * np.dot(coeffs.B, K[:self.n_stages])
            
            # Error estimation - Scipy uses E5 and E3
            err5 = np.dot(K[:self.n_stages+1].T, coeffs.E5) # shape (n,)
            err3 = np.dot(K[:self.n_stages+1].T, coeffs.E3) # shape (n,)
            
            # Scipy's advanced error estimation from Hairer
            denom = np.hypot(np.abs(err5), 0.1 * np.abs(err3))
            correction_factor = np.ones_like(err5)
            mask = denom > 0
            correction_factor[mask] = np.abs(err5[mask]) / denom[mask]
            
            error_est = dt_current * err5 * correction_factor
            
            # Normalize error
            scale = self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y_next))
            error_norm = np.linalg.norm(error_est / scale) / np.sqrt(n)
            
            if error_norm < 1.0:
                # Accept step
                if error_norm == 0:
                    factor = self.max_factor
                else:
                    factor = self.safety_factor * (error_norm ** self.error_exponent)
                    
                dt_new = dt_current * min(self.max_factor, max(self.min_factor, factor))
                
                return y_next, t + dt_current, dt_new
            
            else:
                # Reject step
                factor = self.safety_factor * (error_norm ** self.error_exponent)
                dt_current *= max(self.min_factor, min(1.0, factor))
                
                if abs(dt_current) < 1e-15:
                    raise RuntimeError("Step size too small in RK853")

    def integrate(self, f, t_span, y0, dt=None, **kwargs):
        if dt is None:
            dt = (t_span[1] - t_span[0]) / 100.0
        return super().integrate(f, t_span, y0, dt, **kwargs)
