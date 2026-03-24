"""
Numerical Cowell Propagator.
"""

import numpy as np

from ..integrators.integrator import Integrator
from ..integrators.rk4 import RK4
from .base import Propagator


class CowellPropagator(Propagator):
    """
    Numerical Cowell Propagator.
    Integrates the equations of motion numerically, allowing for perturbations.
    """

    def __init__(self, integrator: Integrator = None, mu=398600.4418e9):
        """
        Initialize the Cowell Propagator.

        Args:
            integrator (Integrator): Numerical integrator instance. Defaults to RK4.
            mu (float): Gravitational parameter [m^3/s^2]. Default is Earth.
        """
        self.integrator = integrator if integrator else RK4()
        self.mu = mu

    def propagate(
        self, r_i: np.ndarray, v_i: np.ndarray, dt: float, perturbation_acc_fn=None, **kwargs
    ):
        """
        Propagates the state using numerical integration.

        Args:
            r_i (np.ndarray): Initial position vector [km] or [m].
            v_i (np.ndarray): Initial velocity vector [km/s] or [m/s].
            dt (float): Propagation duration [s].
            perturbation_acc_fn (callable, optional): Function that returns perturbation accelerations.
                                                      Signature: acc_pert = f(t, r, v)
            dt_step (float, optional): Step size for the integrator. Default is dt (single step) or 10s if dt is large.

        Returns
        -------
            tuple: (r_f, v_f) Final position and velocity vectors.
        """
        # Define the state vector y = [r, v]
        y0 = np.concatenate([r_i, v_i])

        # Define the equations of motion: dy/dt = f(t, y)
        def equations_of_motion(t, y):
            r = y[:3]
            v = y[3:]
            r_mag = np.linalg.norm(r)

            # Two-body acceleration
            a_two_body = -self.mu / (r_mag**3) * r

            # Perturbations
            a_pert = np.zeros(3)
            if perturbation_acc_fn:
                a_pert = perturbation_acc_fn(t, r, v)

            a_total = a_two_body + a_pert

            return np.concatenate([v, a_total])

        # Determine integration step size
        # If the user provides 'dt_step' in kwargs, use it.
        # Integrator.integrate takes an initial 'dt'.
        step_size = kwargs.get("dt_step", 10.0)  # Default 10s step size for numerical integration
        if step_size > dt:
            step_size = dt

        # Integrate
        # We start at t=0 and go to t=dt relative to the epoch of r_i
        t_span = (0, dt)
        t_values, y_values = self.integrator.integrate(
            equations_of_motion, t_span, y0, dt=step_size
        )

        # Extract final state
        y_final = y_values[-1]
        r_f = y_final[:3]
        v_f = y_final[3:]

        return r_f, v_f
