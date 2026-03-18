import pytest
import numpy as np
from gnc_toolkit.guidance.rendezvous import (
    tschauner_hempel_propagation,
    solve_lambert_multi_rev,
    primer_vector_analysis,
    is_within_corridor,
    optimize_rpo_collocation
)

def test_tschauner_hempel_basic():
    x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # 1km radial offset
    oe_target = (7000, 0.01, 0, 0, 0, 0) # a=7000km, e=0.01
    dt = 100.0
    
    pos_f = tschauner_hempel_propagation(x0, oe_target, dt)
    assert pos_f is not None
    assert pos_f.shape == (3,)

def test_lambert_multi_rev_detect():
    r1 = np.array([7000, 0, 0])
    r2 = np.array([-7000, 0, 0])
    dt = 5000.0 # roughly half a period
    
    # Not yet fully implemented mapping, but should raise NotImplementedError instead of crash
    with pytest.raises(NotImplementedError):
        solve_lambert_multi_rev(r1, r2, dt, n_rev=1)

def test_primer_vector_optimality():
    r0 = np.array([7000, 0, 0])
    v0 = np.array([0, 7.546, 0])
    rf = np.array([0, 7000, 0])
    vf = np.array([-7.546, 0, 0])
    dt = 1000.0
    
    res = primer_vector_analysis(r0, v0, rf, vf, dt)
    assert "is_optimal" in res
    assert "dv_total" in res

def test_safe_approach_corridor():
    r_rel = np.array([10.0, 0.1, 0.1])
    axis = np.array([1.0, 0.0, 0.0])
    
    # 5 degree cone
    assert is_within_corridor(r_rel, axis, 5.0) == True
    
    # Way outside
    r_bad = np.array([1.0, 10.0, 0.0])
    assert is_within_corridor(r_bad, axis, 5.0) == False

def test_optimize_collocation_hook():
    r0 = np.array([1.0, 0, 0])
    v0 = np.array([0, 0, 0])
    rf = np.array([0, 0, 0])
    vf = np.array([0, 0, 0])
    dt = 100.0
    
    res = optimize_rpo_collocation(r0, v0, rf, vf, dt)
    assert "success" in res
