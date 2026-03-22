"""
Finite-Horizon Linear Quadratic Regulator (LQR).
"""

import numpy as np
from scipy.integrate import solve_ivp

class FiniteHorizonLQR:
    """
    Finite-Horizon Linear Quadratic Regulator (LQR).
    
    Minimizes:
    J = x(T).T * P_f * x(T) + integral_0^T (x.T * Q(t) * x + u.T * R(t) * u) dt
    
    The optimal control law is u(t) = -K(t) * x(t), 
    where K(t) = R(t)^-1 * B(t).T * P(t)
    and P(t) is the solution to the Differential Riccati Equation (DRE):
    -P_dot = P*A + A.T*P - P*B*R^-1*B.T*P + Q, with P(T) = P_f
    """
    def __init__(self, A_fn, B_fn, Q_fn, R_fn, Pf, T):
        """
        Initialize Finite-Horizon LQR.
        
        Args:
            A_fn (callable): Function A(t) returning nx x nx matrix
            B_fn (callable): Function B(t) returning nx x nu matrix
            Q_fn (callable): Function Q(t) returning nx x nx matrix
            R_fn (callable): Function R(t) returning nu x nu matrix
            Pf (np.ndarray): Final state cost matrix (nx x nx)
            T (float): Final time
        """
        self.A_fn = A_fn
        self.B_fn = B_fn
        self.Q_fn = Q_fn
        self.R_fn = R_fn
        self.Pf = np.array(Pf)
        self.T = T
        self.P_trajectory = None
        self.t_span = None

    def solve(self, num_points=100):
        """
        Solve the Differential Riccati Equation backwards in time.
        """
        nx = self.Pf.shape[0]
        
        def dre(t, p_flat):
            P = p_flat.reshape((nx, nx))
            A = self.A_fn(t)
            B = self.B_fn(t)
            Q = self.Q_fn(t)
            R = self.R_fn(t)
            
            # Using solve for R^-1 * B.T @ P
            K_term = np.linalg.solve(R, B.T @ P)
            P_dot = -(P @ A + A.T @ P - P @ B @ K_term + Q)
            return P_dot.flatten()

        # Solve backwards from T to 0
        t_eval = np.linspace(self.T, 0, num_points)
        sol = solve_ivp(dre, [self.T, 0], self.Pf.flatten(), t_eval=t_eval, method='RK45')
        
        # Store trajectory (reverse it to be 0 to T)
        self.t_span = sol.t[::-1]
        self.P_trajectory = sol.y.T[::-1].reshape((-1, nx, nx))
        
        return self.t_span, self.P_trajectory

    def get_gain(self, t):
        """
        Interpolate the gain matrix K at time t.
        """
        if self.P_trajectory is None:
            self.solve()
            
        # Linear interpolation of P
        nx = self.Pf.shape[0]
        P_t = np.zeros((nx, nx))
        for i in range(nx):
            for j in range(nx):
                P_t[i,j] = np.interp(t, self.t_span, self.P_trajectory[:, i, j])
        
        B = self.B_fn(t)
        R = self.R_fn(t)
        K = np.linalg.solve(R, B.T @ P_t)
        return K

    def compute_control(self, x, t):
        """
        Compute control input u = -K(t) * x
        """
        K = self.get_gain(t)
        return -K @ x
