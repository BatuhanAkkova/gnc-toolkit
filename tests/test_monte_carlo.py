import pytest
from gnc_toolkit.simulation.monte_carlo import MonteCarloSim

class DummySimulator:
    def __init__(self, seed, **kwargs):
        self.seed = seed
        self.kwargs = kwargs

    def run(self):
        return {"seed": self.seed, "param": self.kwargs.get("param")}

def simulator_factory(seed, **kwargs):
    return DummySimulator(seed, **kwargs)

def test_monte_carlo_sequential():
    mc = MonteCarloSim(simulator_factory)
    mc.run_sequential(num_runs=3, param="test")
    
    assert len(mc.results) == 3
    for i in range(3):
        assert mc.results[i] == {"seed": i, "param": "test"}

from unittest.mock import patch, MagicMock

@patch("gnc_toolkit.simulation.monte_carlo.mp.Pool")
def test_monte_carlo_parallel(mock_pool_class):
    mock_pool = MagicMock()
    mock_pool_class.return_value.__enter__.return_value = mock_pool
    
    mc = MonteCarloSim(simulator_factory)
    
    def mock_map(func, iterable):
        return [func(x) for x in iterable]
    mock_pool.map.side_effect = mock_map
    
    mc.run_parallel(num_runs=4, processes=2, param="parallel")
    
    assert len(mc.results) == 4
    results = sorted(mc.results, key=lambda x: x["seed"])
    for i in range(4):
        assert results[i] == {"seed": i, "param": "parallel"}
