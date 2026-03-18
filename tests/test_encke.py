import pytest
import numpy as np
from gnc_toolkit.propagators.encke import EnckePropagator
from gnc_toolkit.propagators.kepler import KeplerPropagator

def test_encke_unperturbed():
    """
    Test Encke propagator with NO perturbations.
    Should match Keplerian propagator exactly (or very closely).
    """
    mu = 398600.4418e9 # m^3/s^2
    r_i = np.array([7000e3, 0.0, 0.0])
    v_i = np.array([0.0, np.sqrt(mu / 7000e3), 0.0])
    dt = 1000.0 # 1000 seconds
    
    encke = EnckePropagator(mu=mu)
    kepler = KeplerPropagator(mu=mu)
    
    r_encke, v_encke = encke.propagate(r_i, v_i, dt)
    r_kep, v_kep = kepler.propagate(r_i, v_i, dt)
    
    # Without perturbations, deviation integrals are zero.
    # Should be identical
    np.testing.assert_allclose(r_encke, r_kep, rtol=1e-6)
    np.testing.assert_allclose(v_encke, v_kep, rtol=1e-6)

def test_encke_perturbed():
    """
    Test Encke propagator with a constant disturbance.
    Verifies it runs without crashing.
    """
    mu = 398600.4418e9
    r_i = np.array([7000e3, 0.0, 0.0])
    v_i = np.array([0.0, np.sqrt(mu / 7000e3), 0.0])
    dt = 100.0
    
    # Constant small thrust/drag
    def constant_disturbance(t, r, v):
        return np.array([1e-6, 0.0, 0.0]) # 1 um/s^2
        
    encke = EnckePropagator(mu=mu, rect_tol=1e-12) # Force rectification
    r_f, v_f = encke.propagate(r_i, v_i, dt, perturbation_acc_fn=constant_disturbance)
    
    assert len(r_f) == 3
    assert len(v_f) == 3
    # Absolute values should diff from unperturbed
    kepler = KeplerPropagator(mu=mu)
    r_kep, _ = kepler.propagate(r_i, v_i, dt)
    
    # Deviation should exist (at least micron level over 100s with 1e-6 acc)
    assert not np.allclose(r_f, r_kep, rtol=0, atol=1e-6)
