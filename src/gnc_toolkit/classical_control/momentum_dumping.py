"""
Reaction wheel momentum desaturation using magnetic torque (Cross-Product Law).
"""

import numpy as np


from typing import Optional, Union, List

class CrossProductLaw:
    r"""
    Reaction wheel momentum desaturation using the Cross-Product Law.

    This controller calculates a magnetic dipole moment 'm' such that the
    resulting magnetic torque $T = m \times B$ opposes the component of the
    angular momentum error perpendicular to the magnetic field.

    Control law: $m = k \frac{H_{err} \times B}{|B|^2}$

    Parameters
    ----------
    gain : float
        Feedback gain $k$ ($k > 0$). Units: $[s^{-1}]$.
    max_dipole : float, optional
        Maximum magnetic dipole moment allowed (saturation) [Am^2].
    """

    def __init__(self, gain: float, max_dipole: Optional[float] = None):
        """Initialize the momentum dumping controller."""
        self.gain = gain
        self.max_dipole = max_dipole

    def calculate_control(
        self, h_error: Union[np.ndarray, List[float]], b_field: Union[np.ndarray, List[float]]
    ) -> np.ndarray:
        """
        Calculate the required magnetic dipole moment.

        Parameters
        ----------
        h_error : np.ndarray or list
            The angular momentum vector to be dumped (3,). Units: [Nms].
        b_field : np.ndarray or list
            The local magnetic field vector in Body frame (3,). Units: [T].

        Returns
        -------
        np.ndarray
            Magnetic dipole moment vector $m$ [Am^2] (3,).
        """
        h_vec = np.asarray(h_error, dtype=float)
        b_vec = np.asarray(b_field, dtype=float)

        b_sq = np.dot(b_vec, b_vec)

        # Singular handling for weak/zero magnetic fields
        if b_sq < 1e-18:
            return np.zeros(3)

        dipole_moment = (self.gain / b_sq) * np.cross(h_vec, b_vec)

        # Apply dipole saturation
        if self.max_dipole is not None:
            norm_m = np.linalg.norm(dipole_moment)
            if norm_m > self.max_dipole:
                dipole_moment *= self.max_dipole / norm_m

        return dipole_moment
