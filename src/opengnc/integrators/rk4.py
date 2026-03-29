"""
Fixed-step Runge-Kutta 4th order integrator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .integrator import Integrator


class RK4(Integrator):
    r"""
    Fixed-step 4th Order Runge-Kutta (RK4).

    Step Logic:
    $k_1 = f(t_n, y_n)$
    $k_2 = f(t_n + \frac{h}{2}, y_n + \frac{h}{2} k_1)$
    $k_3 = f(t_n + \frac{h}{2}, y_n + \frac{h}{2} k_2)$
    $k_4 = f(t_n + h, y_n + h k_3)$
    $y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$

    Parameters
    ----------
    None
    """

    def step(
        self,
        f: Callable,
        t: float,
        y: np.ndarray,
        dt: float,
        **kwargs: Any
    ) -> tuple[np.ndarray, float, float]:
        """
        Perform a single RK4 step.

        Parameters
        ----------
        f : Callable
            Derivative function f(t, y, **kwargs).
        t : float
            Current time $t_n$ (s).
        y : np.ndarray
            Current state $y_n$.
        dt : float
            Time step $h$ (s).

        Returns
        -------
        tuple[np.ndarray, float, float]
            (y_next, t_next, dt).
        """
        y_val = np.asarray(y)

        k1 = np.asarray(f(t, y_val, **kwargs))
        k2 = np.asarray(f(t + 0.5 * dt, y_val + 0.5 * dt * k1, **kwargs))
        k3 = np.asarray(f(t + 0.5 * dt, y_val + 0.5 * dt * k2, **kwargs))
        k4 = np.asarray(f(t + dt, y_val + dt * k3, **kwargs))

        y_next = y_val + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t_next = t + dt

        return y_next, t_next, dt




