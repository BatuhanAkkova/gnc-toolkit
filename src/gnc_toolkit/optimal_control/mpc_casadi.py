"""
Nonlinear Model Predictive Control (NMPC) using CasADi with Multiple Shooting.
"""

import numpy as np
import casadi as ca

class CasadiNMPC:
    """
    Production-grade Nonlinear Model Predictive Controller (NMPC) using CasADi.
    
    Optimizes a finite horizon cost function subject to nonlinear dynamics and constraints.
    Uses Multiple Shooting formulation for numerical stability and faster convergence.
    """
    def __init__(self, nx, nu, horizon, dt, dynamics_func, cost_func, terminal_cost_func,
                 u_min=None, u_max=None, x_min=None, x_max=None, discrete=False):
        """
        Initialize the Casadi NMPC.
        
        Args:
            nx (int): State dimension.
            nu (int): Input dimension.
            horizon (int): Prediction horizon N.
            dt (float): Time step.
            dynamics_func (callable): Function f(x, u) -> x_next (if discrete) or dx/dt (if continuous).
            cost_func (callable): Function L(x, u) -> scalar cost.
            terminal_cost_func (callable): Function V(x) -> scalar terminal cost.
            u_min (float/array): Minimum control input.
            u_max (float/array): Maximum control input.
            x_min (float/array): Minimum state bounds.
            x_max (float/array): Maximum state bounds.
            discrete (bool): If True, dynamics_func returns x_next. If False, returns dx/dt and RK4 integration is used.
        """
        self.nx = int(nx)
        self.nu = int(nu)
        self.N = int(horizon)
        self.dt = float(dt)
        self.f = dynamics_func
        self.L = cost_func
        self.V = terminal_cost_func
        self.discrete = discrete
        
        self.u_min = self._setup_bounds(u_min, self.nu, -np.inf)
        self.u_max = self._setup_bounds(u_max, self.nu, np.inf)
        self.x_min = self._setup_bounds(x_min, self.nx, -np.inf)
        self.x_max = self._setup_bounds(x_max, self.nx, np.inf)
        
        self._setup_solver()

    def _setup_bounds(self, bound, dim, default_val):
        """Helper to convert bounds to continuous array of correct dimension."""
        if bound is None:
            return np.full(dim, default_val)
        bound_arr = np.array(bound).flatten()
        if bound_arr.ndim == 0:
            return np.full(dim, bound_arr)
        if len(bound_arr) == 1:
            return np.full(dim, bound_arr[0])
        if len(bound_arr) != dim:
            raise ValueError(f"Bound dimension mismatch. Expected {dim}, got {len(bound_arr)}")
        return bound_arr

    def _rk4_step(self, x, u):
        """Runge-Kutta 4th order integration step for continuous systems."""
        k1 = self.f(x, u)
        k2 = self.f(x + self.dt / 2.0 * k1, u)
        k3 = self.f(x + self.dt / 2.0 * k2, u)
        k4 = self.f(x + self.dt * k3, u)
        return x + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _setup_solver(self):
        """Formulate the Multiple Shooting NLP and create solver."""
        # Symbolic variables
        X = ca.MX.sym('X', self.nx, self.N + 1)  # State trajectory
        U = ca.MX.sym('U', self.nu, self.N)     # Control trajectory
        X0 = ca.MX.sym('X0', self.nx)           # Parameter: Initial state

        # Objective and Constraints
        obj = 0.0
        g = []  # Inequality/Equality constraints
        
        # Multiple Shooting Equality Constraints
        # X[:,0] = X0
        g.append(X[:, 0] - X0)
        
        for k in range(self.N):
            # Stage cost
            obj += self.L(X[:, k], U[:, k])
            
            # Dynamics propagates X[:,k] -> X_next
            if self.discrete:
                X_next = self.f(X[:, k], U[:, k])
            else:
                X_next = self._rk4_step(X[:, k], U[:, k])
                
            # Shooting constraint: X[:,k+1] must equal propagated X_next
            g.append(X[:, k + 1] - X_next)

        # Terminal cost
        obj += self.V(X[:, self.N])

        # State constraints setup
        # For State constraints inside g, we need to enforce bounds on X directly or via g
        # For multiple shooting, X is a variable, so we set bounds on X directly in variable bounds.
        # So g only contains dynamic equality constraints.
        
        # Flatten variables for solver
        # vars = [X_0, X_1, ..., X_N, U_0, ..., U_{N-1}]
        V_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        g_vars = ca.vertcat(*g)

        # NLP formulation
        nlp = {'x': V_vars, 'f': obj, 'g': g_vars, 'p': X0}

        # Solver options
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-6,
            'ipopt.max_iter': 100
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        # Setup variable bounds lists
        self.lbx = []
        self.ubx = []
        
        # Bounds on X (States)
        # For Multiple shooting, we bound ALL states X_0 to X_N
        for _ in range(self.N + 1):
            self.lbx.extend(self.x_min)
            self.ubx.extend(self.x_max)
            
        # Bounds on U (Controls)
        for _ in range(self.N):
            self.lbx.extend(self.u_min)
            self.ubx.extend(self.u_max)
            
        self.lbx = np.array(self.lbx)
        self.ubx = np.array(self.ubx)
        
        # Bounds on g (Dynamics constraints)
        # All equal to 0 for shooting constraints
        # g contains: X_0 - X0 (nx), X_1 - X_next_0 (nx), ..., X_N - X_next_{N-1} (nx)
        # Total size: nx * (N + 1)
        self.lbg = np.zeros(self.nx * (self.N + 1))
        self.ubg = np.zeros(self.nx * (self.N + 1))

    def solve(self, x0, u_guess=None):
        """
        Solve the NMPC optimization problem.
        
        Args:
            x0 (np.ndarray): Initial state.
            u_guess (np.ndarray, optional): Initial guess for control inputs [N, nu].
            
        Returns:
            np.ndarray: Optimal control sequence [N, nu].
        """
        x0 = np.array(x0).flatten()
        
        # Initial guess for solver variables
        # Set X values to x0, U to u_guess or 0
        x_guess = np.tile(x0, (self.N + 1, 1)).flatten()
        if u_guess is not None:
             u_guess = np.array(u_guess).flatten()
             if len(u_guess) != self.N * self.nu:
                 u_guess = np.zeros(self.N * self.nu)
        else:
             u_guess = np.zeros(self.N * self.nu)
             
        v_guess = np.concatenate([x_guess, u_guess])

        # Call solver
        sol = self.solver(
            x0=v_guess,
            p=x0,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg
        )

        # Extract solution
        v_opt = sol['x'].full().flatten()
        
        # State optimal (skip X0 ... X_N)
        # States are from index 0 to nx*(N+1)
        # Controls are from nx*(N+1) to end
        u_opt_flat = v_opt[self.nx * (self.N + 1):]
        u_opt = u_opt_flat.reshape((self.N, self.nu))
        
        return u_opt
