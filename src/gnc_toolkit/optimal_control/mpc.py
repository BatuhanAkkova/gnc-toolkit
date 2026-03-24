"""
Linear and Nonlinear Model Predictive Control (MPC) solvers.
"""

import numpy as np
from scipy.optimize import minimize


class LinearMPC:
    """
    Linear Model Predictive Controller (MPC).

    Optimizes a finite horizon cost function subject to linear dynamics and constraints.
    Dynamics: x[k+1] = A x[k] + B u[k]
    Cost: sum_{k=0}^{N-1} (x[k].T Q x[k] + u[k].T R u[k]) + x[N].T P x[N]
    """

    def __init__(self, A, B, Q, R, horizon, P=None, u_min=None, u_max=None, x_min=None, x_max=None):
        """
        Initialize the Linear MPC.

        Args:
            A (np.ndarray): State transition matrix (discrete).
            B (np.ndarray): Input matrix (discrete).
            Q (np.ndarray): State cost matrix.
            R (np.ndarray): Input cost matrix.
            horizon (int): Prediction horizon N.
            P (np.ndarray, optional): Terminal cost matrix. Defaults to Q.
            u_min (float/array): Minimum control input.
            u_max (float/array): Maximum control input.
            x_min (float/array): Minimum state bounds.
            x_max (float/array): Maximum state bounds.
        """
        self.A = np.array(A)
        self.B = np.array(B)
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.N = int(horizon)
        self.P = np.array(P) if P is not None else self.Q

        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]

        self.u_min = u_min
        self.u_max = u_max
        self.x_min = x_min
        self.x_max = x_max

    def solve(self, x0):
        """
        Solve the MPC optimization problem.

        Args:
            x0 (np.ndarray): Initial state.

        Returns
        -------
            np.ndarray: Optimal control sequence [u_0, u_1, ..., u_{N-1}].
        """
        x0 = np.array(x0).flatten()

        def objective(U_flat):
            U = U_flat.reshape((self.N, self.nu))
            cost = 0.0
            x = x0.copy()

            for k in range(self.N):
                u = U[k]
                cost += x.T @ self.Q @ x + u.T @ self.R @ u
                x = self.A @ x + self.B @ u

            # Terminal cost
            cost += x.T @ self.P @ x
            return cost

        # Input bounds
        bounds = []
        if self.u_min is not None or self.u_max is not None:
            umin = np.array(self.u_min) if self.u_min is not None else np.full(self.nu, -np.inf)
            umax = np.array(self.u_max) if self.u_max is not None else np.full(self.nu, np.inf)
            if umin.ndim == 0:
                umin = np.full(self.nu, umin)
            if umax.ndim == 0:
                umax = np.full(self.nu, umax)
            umin = umin.flatten()
            umax = umax.flatten()
            if len(umin) != self.nu or len(umax) != self.nu:
                raise ValueError(
                    f"u_min/u_max dimension mismatch. Expected {self.nu}, got {len(umin)}/{len(umax)}"
                )
            for _ in range(self.N):
                for i in range(self.nu):
                    bounds.append((umin[i], umax[i]))
        else:
            bounds = None

        # State constraints (inequality: x_k - x_min >= 0 and x_max - x_k >= 0)
        constraints = []
        xmin = np.array(self.x_min) if self.x_min is not None else np.full(self.nx, -np.inf)
        xmax = np.array(self.x_max) if self.x_max is not None else np.full(self.nx, np.inf)

        if self.x_min is not None or self.x_max is not None:
            if xmin.ndim == 0:
                xmin = np.full(self.nx, xmin)
            if xmax.ndim == 0:
                xmax = np.full(self.nx, xmax)
            xmin = xmin.flatten()
            xmax = xmax.flatten()
            if len(xmin) != self.nx or len(xmax) != self.nx:
                raise ValueError(
                    f"x_min/x_max dimension mismatch. Expected {self.nx}, got {len(xmin)}/{len(xmax)}"
                )

            def state_constraint_fun(U_flat):
                U = U_flat.reshape((self.N, self.nu))
                x = x0.copy()
                cons_values = []
                for k in range(self.N):
                    u = U[k]
                    x = self.A @ x + self.B @ u
                    if self.x_min is not None:
                        cons_values.extend(x - xmin)
                    if self.x_max is not None:
                        cons_values.extend(xmax - x)
                return np.array(cons_values)

            constraints.append({"type": "ineq", "fun": state_constraint_fun})

        U0 = np.zeros(self.N * self.nu)
        res = minimize(objective, U0, method="SLSQP", bounds=bounds, constraints=constraints)

        if not res.success:
            print(f"MPC Warning: Optimization failed: {res.message}")

        optimal_U = res.x.reshape((self.N, self.nu))
        return optimal_U


class NonlinearMPC:
    """
    Nonlinear Model Predictive Controller (NMPC).

    Optimizes a finite horizon cost function subject to nonlinear dynamics.
    Dynamics: x[k+1] = f(x[k], u[k])
    Cost: sum_{k=0}^{N-1} L(x[k], u[k]) + V(x[N])
    """

    def __init__(
        self,
        dynamics_func,
        cost_func,
        terminal_cost_func,
        horizon,
        nx,
        nu,
        u_min=None,
        u_max=None,
        x_min=None,
        x_max=None,
    ):
        """
        Initialize the Nonlinear MPC.

        Args:
            dynamics_func (callable): Function f(x, u) -> x_next
            cost_func (callable): Function L(x, u) -> scalar cost
            terminal_cost_func (callable): Function V(x) -> scalar terminal cost
            horizon (int): Prediction horizon N
            nx (int): State dimension
            nu (int): Input dimension
            u_min (float/array): Minimum control input.
            u_max (float/array): Maximum control input.
            x_min (float/array): Constraints on state
            x_max (float/array): Constraints on state
        """
        self.f = dynamics_func
        self.L = cost_func
        self.V = terminal_cost_func
        self.N = int(horizon)
        self.nx = int(nx)
        self.nu = int(nu)

        self.u_min = u_min
        self.u_max = u_max
        self.x_min = x_min
        self.x_max = x_max

    def solve(self, x0):
        """
        Solve the NMPC optimization problem using single shooting.

        Args:
            x0 (np.ndarray): Initial state.

        Returns
        -------
            np.ndarray: Optimal control sequence [u_0, ..., u_{N-1}].
        """
        x0 = np.array(x0).flatten()

        def objective(U_flat):
            U = U_flat.reshape((self.N, self.nu))
            total_cost = 0.0
            x = x0.copy()

            for k in range(self.N):
                u = U[k]
                total_cost += self.L(x, u)
                x = np.array(self.f(x, u)).flatten()

            total_cost += self.V(x)
            return total_cost

        # Input constraints (Bounds)
        bounds = []
        if self.u_min is not None or self.u_max is not None:
            umin = np.array(self.u_min) if self.u_min is not None else np.full(self.nu, -np.inf)
            umax = np.array(self.u_max) if self.u_max is not None else np.full(self.nu, np.inf)

            if umin.ndim == 0:
                umin = np.full(self.nu, umin)
            if umax.ndim == 0:
                umax = np.full(self.nu, umax)

            umin = umin.flatten()
            umax = umax.flatten()

            if len(umin) != self.nu or len(umax) != self.nu:
                raise ValueError("u_min/u_max mismatch with nu")

            for _ in range(self.N):
                for i in range(self.nu):
                    bounds.append((umin[i], umax[i]))
        else:
            bounds = None

        # State constraints (Nonlinear inequality constraints)
        constraints = []
        if self.x_min is not None or self.x_max is not None:
            xmin = np.array(self.x_min) if self.x_min is not None else np.full(self.nx, -np.inf)
            xmax = np.array(self.x_max) if self.x_max is not None else np.full(self.nx, np.inf)

            if xmin.ndim == 0:
                xmin = np.full(self.nx, xmin)
            if xmax.ndim == 0:
                xmax = np.full(self.nx, xmax)
            xmin = xmin.flatten()
            xmax = xmax.flatten()

            def state_constraint_fun(U_flat):
                U = U_flat.reshape((self.N, self.nu))
                x = x0.copy()
                cons_values = []

                for k in range(self.N):
                    u = U[k]
                    x = np.array(self.f(x, u)).flatten()

                    if self.x_min is not None:
                        cons_values.extend(x - xmin)
                    if self.x_max is not None:
                        cons_values.extend(xmax - x)

                return np.array(cons_values)

            constraints.append({"type": "ineq", "fun": state_constraint_fun})

        U0 = np.zeros(self.N * self.nu)

        res = minimize(objective, U0, method="SLSQP", bounds=bounds, constraints=constraints)

        if not res.success:
            print(f"NMPC Warning: Optimization failed: {res.message}")

        optimal_U = res.x.reshape((self.N, self.nu))
        return optimal_U
