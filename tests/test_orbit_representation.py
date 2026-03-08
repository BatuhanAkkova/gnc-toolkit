import numpy as np
import pytest
from gnc_toolkit.utils.equinoctial_utils import kepler2equinoctial, equinoctial2eci
from gnc_toolkit.utils.mee_utils import kepler2mee, mee2eci
from gnc_toolkit.utils.state_to_elements import eci2kepler

def test_equinoctial_round_trip():
    # LEO orbit
    a = 7000e3
    ecc = 0.01
    incl = np.radians(45)
    raan = np.radians(30)
    argp = np.radians(60)
    M = np.radians(0)
    
    # Kepler to Equinoctial to ECI
    _, h, k, p, q, lm = kepler2equinoctial(a, ecc, incl, raan, argp, M)
    reci, veci = equinoctial2eci(a, h, k, p, q, lm)
    
    # ECI back to Kepler
    a_f, ecc_f, incl_f, raan_f, argp_f, nu_f, M_f, _, _, _, _, _ = eci2kepler(reci, veci)
    
    assert a == pytest.approx(a_f, rel=1e-5)
    assert ecc == pytest.approx(ecc_f, abs=1e-6)
    assert incl == pytest.approx(incl_f, abs=1e-6)

def test_mee_round_trip():
    a = 7000e3
    ecc = 0.01
    incl = np.radians(45)
    raan = np.radians(30)
    argp = np.radians(60)
    nu = np.radians(0)
    
    p, f, g, h, k, L = kepler2mee(a, ecc, incl, raan, argp, nu)
    reci, veci = mee2eci(p, f, g, h, k, L)
    
    a_f, ecc_f, incl_f, raan_f, argp_f, nu_f, _, _, _, _, _, _ = eci2kepler(reci, veci)
    
    assert a == pytest.approx(a_f, rel=1e-5)
    assert ecc == pytest.approx(ecc_f, abs=1e-6)
    assert incl == pytest.approx(incl_f, abs=1e-6)
