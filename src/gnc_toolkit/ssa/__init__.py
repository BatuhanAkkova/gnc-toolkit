"""
Space Situational Awareness (SSA) Module.
"""

from .conjunction import compute_pc_foster, compute_pc_chan
from .tle_interface import TLECatalog, TLEEntity
from .tracking import compute_mahalanobis_distance, correlate_tracks
from .maneuver import plan_avoidance_maneuver

__all__ = [
    'compute_pc_foster',
    'compute_pc_chan',
    'TLECatalog',
    'TLEEntity',
    'compute_mahalanobis_distance',
    'correlate_tracks',
    'plan_avoidance_maneuver'
]
