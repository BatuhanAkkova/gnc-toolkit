import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from gnc_toolkit.utils.euler_utils import euler_to_dcm, dcm_to_euler
from gnc_toolkit.utils.mrp_utils import quat_to_mrp, mrp_to_quat, mrp_to_dcm, get_shadow_mrp
from gnc_toolkit.utils.crp_utils import quat_to_crp, crp_to_quat, crp_addition
from gnc_toolkit.utils.cayley_klein_utils import quat_to_cayley_klein, cayley_klein_to_quat
from gnc_toolkit.utils.state_conversion import quat_to_dcm
from scipy.spatial.transform import Rotation as R

def test_euler_sequences():
    sequences = ['321', '313', '123', '121', '232', '213']
    angles = np.array([0.1, 0.2, 0.3]) # radians
    
    for seq in sequences:
        # Convert to DCM using our tool
        dcm = euler_to_dcm(angles, seq)
        # Convert back
        angles_est = dcm_to_euler(dcm, seq)
        
        np.testing.assert_allclose(angles_est, angles, atol=1e-10)

def test_mrp_conversions():
    # Random quaternion
    q = np.array([0.1, 0.2, 0.3, 0.911])
    q = q / np.linalg.norm(q)
    
    sigma = quat_to_mrp(q)
    q_est = mrp_to_quat(sigma)
    
    np.testing.assert_allclose(q_est, q, atol=1e-10)
    
    # Check DCM consistency
    dcm_mrp = mrp_to_dcm(sigma)
    dcm_quat = quat_to_dcm(q)
    
    # Note: quat_to_dcm might use different convention (Body-to-ECI vs ECI-to-Body)
    # Both our mrp_to_dcm and quat_to_dcm should be consistent
    np.testing.assert_allclose(dcm_mrp, dcm_quat, atol=1e-10)

def test_mrp_shadow():
    sigma = np.array([0.8, 0.0, 0.0])
    sigma_shadow = get_shadow_mrp(sigma)
    
    # MRP and shadow MRP represent same rotation
    q = mrp_to_quat(sigma)
    q_shadow = mrp_to_quat(sigma_shadow)
    
    # Quaternions might be negated but represent same rotation
    if np.dot(q, q_shadow) < 0:
        q_shadow = -q_shadow
        
    np.testing.assert_allclose(q, q_shadow, atol=1e-10)

def test_crp_conversions():
    # Random quaternion (not near 180 deg)
    q = np.array([0.1, 0.1, 0.1, 0.98])
    q = q / np.linalg.norm(q)
    
    q_crp = quat_to_crp(q)
    q_est = crp_to_quat(q_crp)
    
    np.testing.assert_allclose(q_est, q, atol=1e-10)

def test_crp_addition():
    # Two small rotations
    q1_crp = np.array([0.1, 0.0, 0.0])
    q2_crp = np.array([0.0, 0.1, 0.0])
    
    q_res = crp_addition(q1_crp, q2_crp)
    
    # Reference using quaternions
    q1 = crp_to_quat(q1_crp)
    q2 = crp_to_quat(q2_crp)
    
    # Quat multiplication matches rotation composition
    # Result should be q2 * q1 (for Body rotations)
    from gnc_toolkit.utils.quat_utils import quat_mult
    q_ref = quat_mult(q2, q1)
    
    q_est = crp_to_quat(q_res)
    
    if np.dot(q_est, q_ref) < 0:
        q_est = -q_est
        
    np.testing.assert_allclose(q_est, q_ref, atol=1e-10)

def test_cayley_klein():
    q = np.array([0.1, 0.2, 0.3, 0.911])
    q = q / np.linalg.norm(q)
    
    U = quat_to_cayley_klein(q)
    q_est = cayley_klein_to_quat(U)
    
    np.testing.assert_allclose(q_est, q, atol=1e-10)
    
    # Check unitarity
    np.testing.assert_allclose(U @ np.conj(U).T, np.eye(2), atol=1e-10)
