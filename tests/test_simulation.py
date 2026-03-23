import os
import time
import json
import csv
from unittest.mock import patch, MagicMock

import pytest

from gnc_toolkit.simulation.events import EventQueue
from gnc_toolkit.simulation.logging import SimulationLogger
from gnc_toolkit.simulation.simulator import MissionSimulator
from gnc_toolkit.simulation.realtime import RealTimeSimulator

def test_event_queue():
    eq = EventQueue()
    results = []

    def callback1(val):
        results.append(val)

    # Schedule out of order
    eq.schedule(10.0, callback1, "B")
    eq.schedule(5.0, callback1, "A")
    eq.schedule(15.0, callback1, "C")

    assert eq.has_events()
    assert eq.next_event_time() == 5.0

    eq.process_until(7.0)
    assert results == ["A"]
    assert eq.next_event_time() == 10.0

    eq.process_until(20.0)
    assert results == ["A", "B", "C"]
    assert not eq.has_events()

def test_logger(tmp_path):
    log_file = tmp_path / "sim_log"
    logger = SimulationLogger(str(log_file))

    logger.log(0.0, state=[1, 0, 0], measurements=[1.1], estimates=[0.9], commands=[0.1])
    logger.log(1.0, state=[2, 0, 0], measurements=[2.1], estimates=[1.9], commands=[0.2])

    logger.save_json()
    assert (tmp_path / "sim_log.json").exists()
    
    with open(tmp_path / "sim_log.json", "r") as f:
        data = json.load(f)
        assert len(data) == 2
        assert data[0]["time"] == 0.0
        assert data[1]["state"] == [2, 0, 0]

    logger.save_csv()
    assert (tmp_path / "sim_log.csv").exists()
    
    with open(tmp_path / "sim_log.csv", "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 2
        assert "time" in rows[0]
        assert "state" in rows[0]

def dummy_propagator(t, state, dt, control):
    return state + dt + (control if control else 0)

def dummy_sensor(t, state):
    return state + 0.1

def dummy_estimator(t, meas):
    return meas - 0.05

def dummy_controller(t, est):
    return 0.5

def test_mission_simulator():
    sim = MissionSimulator(
        propagator=dummy_propagator,
        sensor_model=dummy_sensor,
        estimator=dummy_estimator,
        controller=dummy_controller
    )
    
    sim.initialize(0.0, initial_state=0.0)
    
    # Schedule an event changing state directly
    def force_state(new_s):
        sim.state = new_s
        
    sim.schedule_event(1.5, force_state, 10.0)
    
    sim.run(3.0, dt=1.0)
    # t=0: state=0 -> u=0.5 -> new state = 0 + 1 + 0.5 = 1.5
    # t=1: state=1.5 -> u=0.5 -> new state = 1.5 + 1 + 0.5 = 3.0
    # t=1.5: event triggers, state = 10.0
    # t=2: state=10.0 -> u=0.5 -> new state = 10.0 + 1 + 0.5 = 11.5
    # t=3: state=11.5 -> u=0.5 -> new state = 11.5 + 1 + 0.5 = 13.0
    
    assert sim.time == 4.0 # 3.0 + dt
    assert sim.state == 13.0

@patch("time.sleep")
@patch("time.time")
def test_realtime_simulator(mock_time, mock_sleep):
    mock_time.side_effect = [0.0, 0.1, 0.2, 0.3] # wall clock
    
    # RTF = 2.0 -> 1s simulation = 0.5s wall clock
    sim = RealTimeSimulator(
        propagator=dummy_propagator,
        sensor_model=None,
        estimator=None,
        controller=None,
        rtf=2.0
    )
    
    sim.initialize(0.0, initial_state=0.0)
    sim.run(1.0, dt=1.0)
    
    # Expected sleep
    mock_sleep.assert_called()
