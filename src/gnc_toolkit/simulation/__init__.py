"""
Simulation module for the gnc_toolkit.

This module provides the architecture for end-to-end mission simulation,
including scenario configuration, discrete-event scheduling, simulation
logging, Monte Carlo harness, and real-time/multi-body support.
"""

from .simulator import MissionSimulator
from .events import Event, EventQueue
from .scenario import ScenarioConfig
from .logging import SimulationLogger
from .monte_carlo import MonteCarloSim
from .realtime import RealTimeSimulator
from .multibody import ConstellationSimulator

__all__ = [
    "MissionSimulator",
    "Event",
    "EventQueue",
    "ScenarioConfig",
    "SimulationLogger",
    "MonteCarloSim",
    "RealTimeSimulator",
    "ConstellationSimulator"
]
