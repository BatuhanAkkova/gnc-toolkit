import multiprocessing as mp
from typing import Callable, List, Dict, Any

class MonteCarloSim:
    """
    Monte Carlo simulation harness for uncertainty quantification and WC analysis.
    """

    def __init__(self, simulator_factory: Callable[..., Any]):
        """
        Initialize the Monte Carlo harness.

        Parameters
        ----------
        simulator_factory : Callable
            A factory function that creates and configures a single simulator instance.
            Signature: `factory_fn(seed: int, kwargs) -> MissionSimulator`
        """
        self.simulator_factory = simulator_factory
        self.results: List[Any] = []

    def _run_single(self, kwargs) -> Any:
        seed = kwargs.pop('seed')
        sim = self.simulator_factory(seed, **kwargs)
        # Assumes sim.run() returns a result object or log
        # if not, `sim.logger.history` might be returned.
        return sim.run()

    def run_sequential(self, num_runs: int, **kwargs):
        """
        Executes simulations sequentially.

        Parameters
        ----------
        num_runs : int
            Number of iterations.
        """
        self.results = []
        for i in range(num_runs):
            params = dict(kwargs)
            params['seed'] = i
            result = self._run_single(params)
            self.results.append(result)

    def run_parallel(self, num_runs: int, processes: int = None, **kwargs):
        """
        Executes simulations in parallel.

        Parameters
        ----------
        num_runs : int
            Number of iterations.
        processes : int, optional
            Number of parallel workers. Defaults to CPU count.
        """
        self.results = []
        pool_kwargs = []
        for i in range(num_runs):
            params = dict(kwargs)
            params['seed'] = i
            pool_kwargs.append(params)
            
        with mp.Pool(processes) as pool:
            self.results = pool.map(self._run_single, pool_kwargs)
