"""
Modified Equinoctial Elements (MEE) kinematics and conversions.
"""

import numpy as np


def kepler2mee(a, ecc, incl, raan, argp, nu, mu=398600.4415e9):
    """
    Converts Keplerian elements to Modified Equinoctial Elements (MEE).
    Elements: p, f, g, h, k, L

    Args:
        a, ecc, incl, raan, argp, nu: Keplerian elements

    Returns
    -------
        tuple: (p, f, g, h, k, L)
    """
    p_mee = a * (1.0 - ecc**2)
    f = ecc * np.cos(raan + argp)
    g = ecc * np.sin(raan + argp)
    h = np.tan(incl / 2.0) * np.cos(raan)
    k = np.tan(incl / 2.0) * np.sin(raan)
    L = raan + argp + nu

    L = np.mod(L, 2 * np.pi)

    return p_mee, f, g, h, k, L


def mee2eci(p, f, g, h, k, L, mu=398600.4415e9):
    """
    Converts Modified Equinoctial Elements to ECI Cartesian state.
    """
    sinL = np.sin(L)
    cosL = np.cos(L)

    w = 1.0 + f * cosL + g * sinL
    r = p / w

    s2 = 1.0 + h**2 + k**2

    reci = (r / s2) * np.array(
        [
            cosL * (1 + h**2 - k**2) + 2 * h * k * sinL,
            sinL * (1 - h**2 + k**2) + 2 * h * k * cosL,
            2 * (h * sinL - k * cosL),
        ]
    )

    v_coeff = np.sqrt(mu / p) / s2
    veci = v_coeff * np.array(
        [
            -(
                sinL * (1 + h**2 - k**2)
                - 2 * h * k * cosL
                + g
                + g * h**2
                - g * k**2
                - 2 * f * h * k
            ),
            (cosL * (1 - h**2 + k**2) - 2 * h * k * sinL + f - f * h**2 + f * k**2 - 2 * g * h * k),
            2 * (cosL + f) * h + 2 * (sinL + g) * k,
        ]
    )

    return reci, veci


def eci2mee(reci, veci, mu=398600.4415e9):
    """
    Converts ECI Cartesian state to Modified Equinoctial Elements.
    """
    from .state_to_elements import eci2kepler

    a, ecc_val, incl_val, raan_val, argp_val, nu_val, _, _, _, _, _, _ = eci2kepler(reci, veci)

    return kepler2mee(a, ecc_val, incl_val, raan_val, argp_val, nu_val, mu)
