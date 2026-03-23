import pytest
import numpy as np
from gnc_toolkit.ssa.conjunction import compute_pc_foster, compute_pc_chan
from gnc_toolkit.ssa.tle_interface import TLECatalog
from gnc_toolkit.ssa.tracking import correlate_tracks, compute_mahalanobis_distance
from gnc_toolkit.ssa.maneuver import plan_avoidance_maneuver

def test_compute_pc_foster_high():
    # Identical position, should have high Pc
    r1 = np.array([7000000.0, 0.0, 0.0])
    v1 = np.array([0.0, 7500.0, 0.0])
    cov1 = np.eye(3) * 100.0 # 10m standard dev
    
    r2 = r1.copy()
    v2 = np.array([0.0, 0.0, 7500.0]) # Cross encounter
    cov2 = np.eye(3) * 100.0
    
    hbr = 20.0 # 20m radius
    
    pc = compute_pc_foster(r1, v1, cov1, r2, v2, cov2, hbr)
    assert pc > 0.1 # Should be high

def test_compute_pc_foster_low():
    # Far apart
    r1 = np.array([7000000.0, 0.0, 0.0])
    v1 = np.array([0.0, 7500.0, 0.0])
    cov1 = np.eye(3) * 100.0
    
    r2 = r1 + np.array([500.0, 0.0, 0.0]) # 500m separation
    v2 = np.array([0.0, 0.0, 7500.0])
    cov2 = np.eye(3) * 100.0
    
    hbr = 10.0
    
    pc = compute_pc_foster(r1, v1, cov1, r2, v2, cov2, hbr)
    assert pc < 0.01

def test_compute_pc_chan_high():
    r1 = np.array([7000000.0, 0.0, 0.0])
    v1 = np.array([0.0, 7500.0, 0.0])
    cov1 = np.eye(3) * 100.0
    r2 = r1.copy()
    v2 = np.array([0.0, 0.0, 7500.0])
    cov2 = np.eye(3) * 100.0
    hbr = 20.0
    pc = compute_pc_chan(r1, v1, cov1, r2, v2, cov2, hbr)
    assert pc > 0.1

def test_compute_pc_chan_low():
    r1 = np.array([7000000.0, 0.0, 0.0])
    v1 = np.array([0.0, 7500.0, 0.0])
    cov1 = np.eye(3) * 100.0
    r2 = r1 + np.array([500.0, 0.0, 0.0])
    v2 = np.array([0.0, 0.0, 7500.0])
    cov2 = np.eye(3) * 100.0
    hbr = 10.0
    pc = compute_pc_chan(r1, v1, cov1, r2, v2, cov2, hbr)
    assert pc < 0.01

def test_compute_pc_compare():
    r1 = np.array([7000000.0, 0.0, 0.0])
    v1 = np.array([0.0, 7500.0, 0.0])
    cov1 = np.eye(3) * 100.0
    r2 = r1 + np.array([50.0, 0.0, 0.0])
    v2 = np.array([0.0, 0.0, 7500.0])
    cov2 = np.eye(3) * 100.0
    hbr = 15.0
    pc_foster = compute_pc_foster(r1, v1, cov1, r2, v2, cov2, hbr)
    pc_chan = compute_pc_chan(r1, v1, cov1, r2, v2, cov2, hbr)
    # Chan is an approximation, but should be close
    assert np.abs(pc_foster - pc_chan) < 0.01

def test_compute_pc_compare_anisotropic():
    r1 = np.array([7000000.0, 0.0, 0.0])
    v1 = np.array([0.0, 7500.0, 0.0])
    cov1 = np.diag([200.0, 50.0, 100.0]) # Anisotropic
    r2 = r1 + np.array([30.0, 20.0, 0.0])
    v2 = np.array([0.0, 0.0, 7500.0])
    cov2 = np.diag([100.0, 50.0, 100.0])
    hbr = 15.0
    pc_foster = compute_pc_foster(r1, v1, cov1, r2, v2, cov2, hbr)
    pc_chan = compute_pc_chan(r1, v1, cov1, r2, v2, cov2, hbr)
    assert np.abs(pc_foster - pc_chan) < 0.015

def test_tle_catalog():
    cat = TLECatalog()
    # Sample ISS TLE (just placeholder strings for format)
    name = "ISS (ZARYA)"
    l1 = "1 25544U 98067A   26081.56250000  .00016717  00000-0  10270-3 0  9011"
    l2 = "2 25544  51.6416 242.3456 0005321 123.4567 245.6789 15.49123456123450"
    
    cat.add_tle(name, l1, l2)
    assert "ISS (ZARYA)" in cat.list_satellites()
    
    sat = cat.get_by_norad_id("25544")
    assert sat is not None
    assert sat.name == "ISS (ZARYA)"
    
    sat_by_name = cat.get_by_name("ISS (ZARYA)")
    assert sat_by_name is not None

def test_correlate_tracks():
    x1 = np.zeros(6)
    x2 = np.zeros(6)
    cov1 = np.eye(6)
    cov2 = np.eye(6)
    
    assert correlate_tracks(x1, x2, cov1, cov2, threshold=3.0) == True
    
    x2[0] = 5.0 # 5 sigma away
    assert correlate_tracks(x1, x2, cov1, cov2, threshold=3.0) == False

def test_plan_avoidance_maneuver():
    r_sat = np.array([7000000.0, 0.0, 0.0])
    v_sat = np.array([0.0, 7500.0, 0.0])
    
    r_debris = r_sat + np.array([5.0, 0.0, 0.0]) # 5m separation
    v_debris = np.array([0.0, 0.0, 7500.0])
    
    safety_radius = 50.0
    t_encounter = 100.0 # 100s to encounter
    
    dv, est_miss = plan_avoidance_maneuver(r_sat, v_sat, r_debris, v_debris, safety_radius, t_encounter)
    
    assert np.linalg.norm(dv) > 0.0
    assert est_miss >= safety_radius
