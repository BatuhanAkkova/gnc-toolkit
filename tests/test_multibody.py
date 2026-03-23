import pytest
from gnc_toolkit.simulation.multibody import ConstellationSimulator
from gnc_toolkit.simulation.simulator import MissionSimulator

def dummy_propagator(t, state_list, dt, control):
    # Propagate multiple states
    return [s + dt for s in state_list]

def test_constellation_simulator():
    sim = ConstellationSimulator(
        num_satellites=3,
        propagator=dummy_propagator
    )
    
    with pytest.raises(ValueError):
        sim.initialize(0.0, initial_states=[0.0, 0.0]) # Wrong number of states
        
    sim.initialize(0.0, initial_states=[1.0, 2.0, 3.0])
    
    assert sim.simulator.time == 0.0
    assert sim.simulator.state == [1.0, 2.0, 3.0]
    
    sim.run(2.0, dt=1.0)
    
    # Needs two steps of dt=1.0
    # Step 1: state = [2.0, 3.0, 4.0]
    # Step 2: state = [3.0, 4.0, 5.0]
    assert sim.simulator.time == 3.0
    assert sim.simulator.state == [3.0, 4.0, 5.0]
