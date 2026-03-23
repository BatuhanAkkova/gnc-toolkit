import numpy as np
import pytest
from gnc_toolkit.utils.mean_elements import osculating2mean, get_j2_secular_rates

def test_osculating2mean_identity():
    # Currently a zero-order approx (returns same)
    elements = (7000e3, 0.01, 1.0, 0.0, 0.0, 0.0)
    out = osculating2mean(*elements)
    assert out == elements

def test_j2_secular_rates():
    a = 7000e3
    ecc = 0.001
    incl = np.radians(45.0)
    
    raan_dot, argp_dot, M_dot = get_j2_secular_rates(a, ecc, incl)
    
    # Verify non-zero and correct signs for typical retro/prograde node drift
    # cos(45) > 0 -> raan_dot should be negative (westward drift)
    assert raan_dot < 0
    
    # 4 - 5*sin^2(45) = 4 - 5*(0.5) = 1.5 > 0 -> argp_dot should be positive
    assert argp_dot > 0
    
    # M_dot should be > n (mean motion)
    n = np.sqrt(398600.4415e9 / a**3)
    assert M_dot > n
