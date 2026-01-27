import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scipy.spatial.transform import Rotation as R
from src.attitude_determination.triad import triad
from src.attitude_determination.quest import quest
from src.utils.quat_utils import quat_to_rmat, quat_mult, quat_inv

def test_triad_identity():
    # Case 1: Identity rotation
    r1 = np.array([1, 0, 0])
    r2 = np.array([0, 1, 0])
    b1 = r1
    b2 = r2
    
    # R_BI should be identity
    R_est = triad(b1, b2, r1, r2)
    np.testing.assert_allclose(R_est, np.eye(3), atol=1e-6)

def test_triad_90deg_z():
    # Case 2: 90 deg rotation about Z axis
    # Body frame is rotated 90 deg wrt Inertial
    
    rot = R.from_euler('z', 90, degrees=True)
    R_BI_true = rot.as_matrix()
    
    r1 = np.array([1, 0, 0])
    r2 = np.array([0, 1, 0])
    
    b1 = R_BI_true @ r1
    b2 = R_BI_true @ r2
    
    R_est = triad(b1, b2, r1, r2)
    
    np.testing.assert_allclose(R_est, R_BI_true, atol=1e-6)

def test_quest_simple():
    # Case: Random rotation, 3 vectors
    rot = R.from_euler('xyz', [30, 45, 60], degrees=True)
    R_BI_true = rot.as_matrix()
    q_true_scipy = rot.as_quat() # [x, y, z, w]
    
    # Create vectors
    r_vecs = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ])
    b_vecs = []
    for r in r_vecs:
        # Calculate body vectors
        b = R_BI_true @ r
        b_vecs.append(b)
    b_vecs = np.array(b_vecs)
    
    q_est = quest(b_vecs, r_vecs)
    
    # Quaternion sign ambiguity
    # q and -q represent same rotation
    if np.dot(q_est, q_true_scipy) < 0:
        q_est = -q_est
        
    np.testing.assert_allclose(q_est, q_true_scipy, atol=1e-6)

def test_triad_collinear_error():
    v = np.array([1, 0, 0])
    with pytest.raises(ValueError):
        triad(v, v, v, v)

def test_quest_mismatch_error():
    b_vecs = np.zeros((3, 3))
    r_vecs = np.zeros((2, 3))
    with pytest.raises(ValueError):
        quest(b_vecs, r_vecs)
