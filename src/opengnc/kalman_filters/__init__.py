from .akf import AKF
from .ckf import CKF
from .ekf import EKF
from .enkf import EnKF
from .imm import IMM
from .kf import KF
from .mekf import MEKF
from .pf import ParticleFilter
from .rts_smoother import rts_smoother
from .sr_ukf import SRUKF
from .ukf import UKF, UKF_Attitude

# Save original Python versions
PythonMEKF = MEKF
PythonUKF_Attitude = UKF_Attitude

# Accelerated C++ Implementations (if available)
ACCELERATION_AVAILABLE = False
try:
    import sys
    import os
    # Add common build paths to search for the extension
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    
    # Check multiple build locations
    build_paths = [
        os.path.join(project_root, 'build', 'Release'),
        os.path.join(project_root, 'build'),
        os.path.join(project_root, 'cpp', 'build', 'Release'),
        os.path.join(project_root, 'cpp', 'build')
    ]
    
    for path in build_paths:
        if os.path.exists(path):
            sys.path.append(path)

    import opengnc_py
    
    # Swap public classes with accelerated ones
    MEKF = opengnc_py.MEKF
    UKF_Attitude = opengnc_py.UKF_Attitude
    ACCELERATION_AVAILABLE = True
    
    # print("[OpenGNC] C++ acceleration enabled for filters: MEKF, UKF_Attitude")
except ImportError:
    pass




