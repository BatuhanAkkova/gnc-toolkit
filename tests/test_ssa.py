import pytest
import numpy as np
from opengnc.ssa.conjunction import compute_pc_foster, compute_pc_chan
from opengnc.ssa.tle_interface import TLECatalog, TLEEntity
from opengnc.ssa.tracking import correlate_tracks, compute_mahalanobis_distance
from opengnc.ssa.maneuver import plan_avoidance_maneuver

def test_compute_pc_foster_high():
    r1 = np.array([7000000.0, 0.0, 0.0])
    v1 = np.array([0.0, 7500.0, 0.0])
    cov1 = np.eye(3) * 100.0
    
    r2 = r1.copy()
    v2 = np.array([0.0, 0.0, 7500.0]) # Cross encounter
    cov2 = np.eye(3) * 100.0
    
    hbr = 20.0 # 20m radius
    
    pc = compute_pc_foster(r1, v1, cov1, r2, v2, cov2, hbr)
    assert pc > 0.1 # Should be high

def test_compute_pc_foster_low():
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

def test_compute_pc_foster_edge_cases():
    r1 = np.array([7000000.0, 0.0, 0.0])
    v1 = np.array([0.0, 0.0, 0.0])
    cov1 = np.eye(3)
    r2 = r1.copy()
    v2 = np.array([0.0, 0.0, 0.0])
    cov2 = np.eye(3)
    with pytest.raises(ValueError, match="Relative velocity is too small"):
        compute_pc_foster(r1, v1, cov1, r2, v2, cov2, 10.0)

    v1_aligned = np.array([1.0, 0.0, 0.0])
    v2_aligned = np.array([0.0, 0.0, 0.0])
    r1_aligned = np.array([1.0, 1.0, 0.0])
    r2_aligned = np.array([0.0, 0.0, 0.0])
    pc = compute_pc_foster(r1_aligned, v1_aligned, cov1, r2_aligned, v2_aligned, cov2, 10.0)
    assert np.isfinite(pc)

    r1_tca = np.array([0.0, 0.0, 0.0])
    r2_tca = np.array([0.0, 0.0, 0.0])
    pc_tca = compute_pc_foster(r1_tca, v1_aligned, cov1, r2_tca, v2_aligned, cov2, 10.0)
    assert np.isfinite(pc_tca)

    cov_singular = np.zeros((3,3))
    pc_singular = compute_pc_foster(r1, v1_aligned, cov_singular, r2, v2_aligned, cov_singular, 10.0)
    assert pc_singular == 0.0

def test_compute_pc_chan_edge_cases():
    r1 = np.array([7000000.0, 0.0, 0.0])
    v1 = np.array([0.0, 0.0, 0.0])
    cov1 = np.eye(3)
    r2 = r1.copy()
    v2 = np.array([0.0, 0.0, 0.0])
    cov2 = np.eye(3)
    with pytest.raises(ValueError, match="Relative velocity is too small"):
        compute_pc_chan(r1, v1, cov1, r2, v2, cov2, 10.0)

    v1_aligned = np.array([1.0, 0.0, 0.0])
    v2_aligned = np.array([0.0, 0.0, 0.0])
    r1_aligned = np.array([1.0, 1.0, 0.0])
    r2_aligned = np.array([0.0, 0.0, 0.0])
    pc = compute_pc_chan(r1_aligned, v1_aligned, cov1, r2_aligned, v2_aligned, cov2, 10.0)
    assert np.isfinite(pc)

    r1_tca = np.array([0.0, 0.0, 0.0])
    r2_tca = np.array([0.0, 0.0, 0.0])
    pc_tca = compute_pc_chan(r1_tca, v1_aligned, cov1, r2_tca, v2_aligned, cov2, 10.0)
    assert np.isfinite(pc_tca)

    cov_singular = np.zeros((3,3))
    pc_singular = compute_pc_chan(r1, v1_aligned, cov_singular, r2, v2_aligned, cov_singular, 10.0)
    assert pc_singular == 0.0

def test_plan_avoidance_maneuver_edge_cases():
    r_sat = np.array([7000000.0, 0.0, 0.0])
    v_sat = np.array([0.0, 0.0, 0.0])
    r_debris = r_sat.copy()
    v_debris = v_sat.copy()
    
    with pytest.raises(ValueError, match="Velocity is too small"):
        plan_avoidance_maneuver(r_sat, v_sat, r_debris, v_debris, 10.0, 100.0)

    v_sat_ok = np.array([0.0, 7500.0, 0.0])
    r_debris_far = r_sat + np.array([100.0, 0.0, 0.0])
    dv, miss = plan_avoidance_maneuver(r_sat, v_sat_ok, r_debris_far, v_debris, 50.0, 100.0)
    assert np.all(dv == 0)

    r_debris_near = r_sat + np.array([10.0, 0.0, 0.0])
    dv, miss = plan_avoidance_maneuver(r_sat, v_sat_ok, r_debris_near, v_debris, 50.0, 5.0)
    assert np.linalg.norm(dv) > 0

def test_tle_interface_edge_cases(tmp_path):
    e = TLEEntity("Test", "1", "2")
    assert e.norad_id == "00000"

    l1 = "1 25544U 98067A   26081.56250000  .00016717  00000-0  10270-3 0  9011"
    l2 = "2 25544  51.6416 242.3456 0005321 123.4567 245.6789 15.49123456123450"
    e2 = TLEEntity("ISS", l1, l2)
    prop = e2.get_propagator()
    assert prop is not None

    filepath = tmp_path / "test_tle.txt"
    content = """INVALID_LINE
ISS (ZARYA)
1 25544U 98067A   26081.56250000  .00016717  00000-0  10270-3 0  9011
2 25544  51.6416 242.3456 0005321 123.4567 245.6789 15.49123456123450
1 12345U 98067A   26081.56250000  .00016717  00000-0  10270-3 0  9011
2 12345  51.6416 242.3456 0005321 123.4567 245.6789 15.49123456123450
DUMMY_LINE
DUMMY_LINE
"""

    filepath.write_text(content)
    cat = TLECatalog()
    cat.load_from_txt(str(filepath))
    assert "ISS (ZARYA)" in cat.list_satellites()
    assert "SAT_12345" in cat.list_satellites()

def test_compute_mahalanobis_distance_singular():
    x1 = np.zeros(6)
    x2 = np.ones(6)
    cov1 = np.zeros((6,6))
    cov2 = np.zeros((6,6))
    
    dist = compute_mahalanobis_distance(x1, x2, cov1, cov2)
    assert np.isclose(dist, np.linalg.norm(x1 - x2))

def test_conjunction_coverage():
    r1 = np.array([7000e3, 0, 0])
    v1 = np.array([0, 7500, 0])
    r2 = r1 + np.array([1, 0.0000000001, 0])
    v2 = v1 + np.array([7500, 0, 0])
    hbr = 10.0
    cov = np.eye(3)
    pc = compute_pc_foster(r1, v1, cov, r2, v2, cov, hbr)
    assert pc >= 0




