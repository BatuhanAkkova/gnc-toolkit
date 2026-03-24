"""
Encke's Method Propagator.
"""

import numpy as np

from ..integrators.integrator import Integrator
from ..integrators.rk4 import RK4
from .base import Propagator
from .kepler import KeplerPropagator


class EnckePropagator(Propagator):
    """
    Encke's Method Propagator.
    Integrates the deviation from a reference Keplerian orbit.
    Suitable for orbit propagation with small disturbances.
    """

    def __init__(self, integrator: Integrator = None, mu=398600.4418e9, rect_tol=1e-6):
        """
        Initialize EnckePropagator.

        Args:
            integrator (Integrator): Numerical integrator for deviations. Defaults to RK4.
            mu (float): Gravitational parameter [m^3/s^2].
            rect_tol (float): Rectification tolerance (|dr| / r_ref). Default 1e-6.
        """
        self.integrator = integrator if integrator else RK4()
        self.mu = mu
        self.rect_tol = rect_tol
        self.kepler = KeplerPropagator(mu=mu)

    def propagate(
        self, r_i: np.ndarray, v_i: np.ndarray, dt: float, perturbation_acc_fn=None, **kwargs
    ):
        """
        Propagate state using Encke's method.

        Args:
            r_i (np.ndarray): Initial position vector [m].
            v_i (np.ndarray): Initial velocity vector [m/s].
            dt (float): Propagation duration [s].
            perturbation_acc_fn (callable, optional): Function that returns perturbation accelerations.
                                                      Signature: acc_pert = f(t, r, v)
        """
        step_size = kwargs.get("dt_step", 10.0)
        if step_size > dt:
            step_size = dt

        curr_t = 0.0
        curr_r_ref = np.array(r_i, dtype=float)
        curr_v_ref = np.array(v_i, dtype=float)
        curr_y_dev = np.zeros(6, dtype=float)  # [dr, dv]

        while curr_t < dt:
            h = step_size
            if curr_t + h > dt:
                h = dt - curr_t

            # Define local EOM relative to current reference state
            def local_eom(t_local, y):
                d_r = y[:3]
                d_v = y[3:]

                # Analytical reference state propagation
                r_r, v_r = self.kepler.propagate(curr_r_ref, curr_v_ref, t_local)
                r_r_mag = np.linalg.norm(r_r)

                r_tot = r_r + d_r
                v_tot = v_r + d_v

                # Encke param: q = dr . (dr - 2 r_ref) / r_ref^2
                q_val = np.dot(d_r, d_r - 2 * r_r) / (r_r_mag**2)

                # f(q) = q * (3 + 3q + q^2) / (1 + (1+q)^(3/2))
                with np.errstate(divide="ignore", invalid="ignore"):
                    if abs(q_val) < 1e-12:
                        f_q_val = 0.0  # Small q limit
                    else:
                        f_q_val = q_val * (3 + 3 * q_val + q_val**2) / (1 + (1 + q_val) ** 1.5)

                a_pt = np.zeros(3)
                if perturbation_acc_fn:
                    a_pt = perturbation_acc_fn(curr_t + t_local, r_tot, v_tot)

                a_en = a_pt + (self.mu / (r_r_mag**3)) * (f_q_val * r_r - d_r)
                return np.concatenate([d_v, a_en])

            # Advance deviation step
            next_y_dev, _, _ = self.integrator.step(local_eom, 0, curr_y_dev, h)
            curr_t += h

            # Advance reference analytically to checkout rectification
            curr_r_ref_next, curr_v_ref_next = self.kepler.propagate(curr_r_ref, curr_v_ref, h)
            dr_next = next_y_dev[:3]

            r_tot_next = curr_r_ref_next + dr_next
            r_tot_next_mag = np.linalg.norm(r_tot_next)

            # Check Rectification Condition
            if np.linalg.norm(dr_next) / r_tot_next_mag > self.rect_tol:
                # Rectify
                curr_r_ref = r_tot_next
                curr_v_ref = curr_v_ref_next + next_y_dev[3:]
                curr_y_dev = np.zeros(6, dtype=float)
            else:
                # No rectification
                curr_r_ref = curr_r_ref_next
                curr_v_ref = curr_v_ref_next
                curr_y_dev = next_y_dev

        # Final state
        r_f = curr_r_ref + curr_y_dev[:3]
        v_f = curr_v_ref + curr_y_dev[3:]

        return r_f, v_f
