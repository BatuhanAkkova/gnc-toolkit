from typing import List, Dict, Any, Callable
from .simulator import MissionSimulator
from .logging import SimulationLogger

class ConstellationSimulator:
    """
    Multi-body / constellation simulator class.
    Simulates N independent or coupled spacecraft simultaneously.
    """

    def __init__(self, 
                 num_satellites: int,
                 propagator: Callable,
                 sensor_model: Callable = None,
                 estimator: Callable = None,
                 controller: Callable = None,
                 logger: SimulationLogger = None):
        """
        Initialize constellation simulator.

        Parameters
        ----------
        num_satellites : int
            Number of spacecraft in the simulation.
        propagator : Callable
            Propagator that can advance multiple states or take a joint state vector.
        sensor_model : Callable
            Global sensor model.
        estimator : Callable
            Global estimator (e.g. tracking multiline).
        controller : Callable
            Global controller coordinating formation/constellation.
        """
        self.num_satellites = num_satellites
        self.simulator = MissionSimulator(
            propagator=propagator,
            sensor_model=sensor_model,
            estimator=estimator,
            controller=controller,
            logger=logger
        )

    def initialize(self, t0: float, initial_states: List[Any]):
        """
        Initialize constellation simulation.

        Parameters
        ----------
        t0 : float
            Start time.
        initial_states : List[Any]
            List of initial states for each spacecraft.
        """
        if len(initial_states) != self.num_satellites:
            raise ValueError(f"Expected {self.num_satellites} initial states, got {len(initial_states)}")
            
        self.simulator.initialize(t0, initial_states)

    def run(self, t_end: float, dt: float):
        """
        Runs the full constellation simulation.

        Parameters
        ----------
        t_end : float
            Stop time.
        dt : float
            Time step.
        """
        self.simulator.run(t_end, dt)
