from typing import Any, Callable

from .events import EventQueue
from .logging import SimulationLogger

class MissionSimulator:
    """
    End-to-end mission simulator class.
    Executes a unified simulation loop: propagate -> sense -> estimate -> control.
    """

    def __init__(self, 
                 propagator: Callable,
                 sensor_model: Callable,
                 estimator: Callable,
                 controller: Callable,
                 logger: SimulationLogger = None):
        """
        Initialize the mission simulator.

        Parameters
        ----------
        propagator : Callable
            Function/method to advance the true state.
            Signature should be `propagator(t, state, dt, control)` -> `new_state`
        sensor_model : Callable
            Function/method to generate measurements from truth.
            Signature should be `sensor_model(t, state)` -> `measurements`
        estimator : Callable
            Function/method to update the state estimate.
            Signature should be `estimator(t, measurements)` -> `state_estimate`
        controller : Callable
            Function/method to calculate control inputs.
            Signature should be `controller(t, state_estimate)` -> `control`
        logger : SimulationLogger, optional
            Logger instance to record the simulation trace.
        """
        self.propagator = propagator
        self.sensor_model = sensor_model
        self.estimator = estimator
        self.controller = controller
        
        self.logger = logger
        self.event_queue = EventQueue()
        
        self.time = 0.0
        self.state = None

    def initialize(self, t0: float, initial_state: Any):
        """
        Sets the initial condition for the simulation.

        Parameters
        ----------
        t0 : float
            Start simulation time.
        initial_state : Any
            Initial truth system state.
        """
        self.time = t0
        self.state = initial_state

    def schedule_event(self, t: float, callback: Callable, *args, **kwargs):
        """
        Schedules a discrete event in the simulation.
        
        Parameters
        ----------
        t : float
            Event execution time.
        callback : Callable
            Function to call.
        """
        self.event_queue.schedule(t, callback, *args, **kwargs)

    def step(self, dt: float):
        """
        Executes a single simulation step.

        Parameters
        ----------
        dt : float
            Time step duration.
        """
        # 1. Process discrete events up to current time
        self.event_queue.process_until(self.time)

        # 2. Sense
        meas = self.sensor_model(self.time, self.state) if self.sensor_model else None

        # 3. Estimate
        est = self.estimator(self.time, meas) if self.estimator else None

        # 4. Control
        # If no estimator is provided, feed the true state (perfect knowledge)
        ctrl_input = est if est is not None else self.state
        u = self.controller(self.time, ctrl_input) if self.controller else None

        # 5. Log variables
        if self.logger:
            self.logger.log(self.time, self.state, meas, est, u)

        # 6. Propagate truth
        self.state = self.propagator(self.time, self.state, dt, u) if self.propagator else self.state
        self.time += dt

    def run(self, t_end: float, dt: float):
        """
        Runs the simulation loop until the end time.

        Parameters
        ----------
        t_end : float
            Simulation end time.
        dt : float
            Simulation time step.
        """
        while self.time <= t_end:
            self.step(dt)
