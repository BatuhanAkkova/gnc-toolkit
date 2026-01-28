import numpy as np
from .integrator import Integrator

class RK45(Integrator):
    """
    Runge-Kutta-Fehlberg 4(5) Variable Step Integrator.
    """
    
    def __init__(self, rtol=1e-6, atol=1e-9, safety_factor=0.9, min_factor=0.2, max_factor=10.0):
        self.rtol = rtol
        self.atol = atol
        self.safety_factor = safety_factor
        self.min_factor = min_factor
        self.max_factor = max_factor
        
        self.c = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
        self.a = [
            [],
            [1/4],
            [3/32, 9/32],
            [1932/2197, -7200/2197, 7296/2197],
            [439/216, -8, 3680/513, -845/4104],
            [-8/27, 2, -3544/2565, 1859/4104, -11/40]
        ]
        # 4th order solution (y)
        # 4th order solution (y)
        self.b4 = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])
        # 5th order solution (z for error est)
        self.b5 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
        
        self.E = self.b5 - self.b4 # Error coefficients


    def step(self, f, t, y, dt, **kwargs):
        """
        Perform a single adaptive RK45 step.
        """
        # Start loop to find acceptable step
        dt_current = dt
        
        while True:
            k = []
            # Calculate k stages
            # k1
            k.append(f(t, y, **kwargs))
            
            # k2..k6
            for i in range(1, 6):
                # sum(a_ij * kj)
                # y_arg = y + dt * sum(...)
                sum_ak = np.zeros_like(y)
                for j in range(i):
                    sum_ak += self.a[i][j] * k[j]
                
                k.append(f(t + self.c[i] * dt_current, y + dt_current * sum_ak, **kwargs))
            
            k = np.array(k) # shape (6, dim)
            
            # Calculate 5th order approximation
            y_next = y + dt_current * np.dot(self.b5, k)
            
            # Calculate error estimate
            # error = dt * sum((b5 - b4) * k)
            error_est = dt_current * np.dot(self.E, k)
            scale = self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y_next))
            error_ratio = np.max(np.abs(error_est) / scale)
            
            if error_ratio < 1.0:
                # Step accepted
                t_next = t + dt_current
                
                # Propose next step size
                if error_ratio < 1.0e-10:
                    dt_new = dt_current * self.max_factor
                else:
                    dt_new = dt_current * self.safety_factor * (error_ratio ** -0.2)
                    
                dt_new = max(dt_current * self.min_factor, min(dt_new, dt_current * self.max_factor))
                
                return y_next, t_next, dt_new
            else:
                # Step rejected, reduce step size
                dt_current = dt_current * self.safety_factor * (error_ratio ** -0.25)
                # Avoid infinite reduction
                if abs(dt_current) < 1e-15:
                    raise RuntimeError(f"Step size too small in RK45: {dt_current}")

    def integrate(self, f, t_span, y0, dt=None, **kwargs):
        """
        Integrate over time span. Overridden to handle initial dt.
        """
        if dt is None:
            dt = (t_span[1] - t_span[0]) / 100.0
            
        return super().integrate(f, t_span, y0, dt, **kwargs)
