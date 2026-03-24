"""
SGP4/SDP4 Analytical Propagator.
"""

import numpy as np
from sgp4.api import Satrec

from .base import Propagator


class Sgp4Propagator(Propagator):
    """
    SGP4/SDP4 Analytical Propagator.
    Wraps the `sgp4` library to propagate orbits using Two-Line Elements (TLEs).
    State output is typically in the TEME frame.
    """

    def __init__(self, line1: str, line2: str):
        """
        Initialize the SGP4 Propagator with a TLE.

        Args:
            line1 (str): First line of the TLE.
            line2 (str): Second line of the TLE.
        """
        self.sat = Satrec.twoline2rv(line1, line2)
        # TLE Epoch Julian Date
        self.jdsatepoch = self.sat.jdsatepoch
        self.jdsatepochF = self.sat.jdsatepochF

    def propagate(self, r_i: np.ndarray, v_i: np.ndarray, dt: float, **kwargs):
        """
        Propagates the satellite state from TLE epoch forward by dt.
        Note: initial position/velocity input (r_i, v_i) is IGNORED,
        as SGP4 is fully determined by the initialized TLE.

        Args:
            r_i (np.ndarray): Ignored.
            v_i (np.ndarray): Ignored.
            dt (float): Propagation time relative to TLE epoch [s].

        Returns
        -------
            tuple: (r_f, v_f) Position and Velocity in TEME frame [m, m/s].
        """
        # SGP4 takes time in minutes from epoch
        minutes_from_epoch = dt / 60.0

        # sgp4 evaluation
        # returns (error_code, position, velocity)
        # position in km, velocity in km/s (TEME)
        err, r_km, v_kms = self.sat.sgp4(
            self.jdsatepoch, self.jdsatepochF + minutes_from_epoch / 1440.0
        )

        if err != 0:
            raise RuntimeError(f"SGP4 error code {err}: Propagation failed.")

        # Convert to SI units [m, m/s]
        r_f = np.array(r_km) * 1000.0
        v_f = np.array(v_kms) * 1000.0

        return r_f, v_f

    def propagate_to_jd(self, jd_f: float, jd_f_frac: float = 0.0):
        """
        Propagate to a specific Julian Date.
        Returns state in TEME [m, m/s].
        """
        err, r_km, v_kms = self.sat.sgp4(jd_f, jd_f_frac)
        if err != 0:
            raise RuntimeError(f"SGP4 error code {err}")
        return np.array(r_km) * 1000.0, np.array(v_kms) * 1000.0
