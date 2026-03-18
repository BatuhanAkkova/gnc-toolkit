"""
Fault Detection, Isolation & Recovery (FDIR) module.
"""

from .residual_generation import ObserverResidualGenerator, AnalyticalRedundancy
from .parity_space import ParitySpaceDetector
from .safe_mode import SafeModeLogic, SafeModeCondition
from .failure_accommodation import ActuatorAccommodation

__all__ = [
    "ObserverResidualGenerator",
    "AnalyticalRedundancy",
    "ParitySpaceDetector",
    "SafeModeLogic",
    "SafeModeCondition",
    "ActuatorAccommodation",
]
