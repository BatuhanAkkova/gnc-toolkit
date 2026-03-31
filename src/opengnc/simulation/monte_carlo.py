import multiprocessing as mp
from collections.abc import Callable
from typing import Any
from .monte_carlo_analyzer import MonteCarloAnalyzer


class MonteCarloSim:
    """
    Monte Carlo Simulation Harness.

    Facilitates large-scale robustness analysis by executing multiple 
    simulation runs with stochastic variations.

    Parameters
    ----------
    simulator_factory : Callable[..., MissionSimulator]
        Generator function to produce specialized simulator instances.
        Signature: `(seed, **kwargs) -> MissionSimulator`.
    """

    def __init__(self, simulator_factory: Callable[..., Any]) -> None:
        """Initialize the harness with a simulator factory."""
        self.simulator_factory = simulator_factory
        self.results: list[Any] = []

    def _run_single(self, kwargs: dict[str, Any]) -> Any:
        """Internal worker for a single Monte Carlo trial."""
        seed = kwargs.pop("seed")
        sim = self.simulator_factory(seed, **kwargs)
        # If the factory returns an object with a run() method, use it.
        # Otherwise, assume the factory returned the result directly.
        if hasattr(sim, "run") and callable(sim.run):
            return sim.run()
        return sim

    def run_sequential(self, num_runs: int, **kwargs: Any) -> list[Any]:
        """
        Execute Monte Carlo iterations in a single thread.

        Parameters
        ----------
        num_runs : int
            Number of trials to execute.
        **kwargs
            Variable parameters passed to the simulator factory.

        Returns
        -------
        List[Any]
            Aggregated results from all trials.
        """
        self.results = []
        for i in range(num_runs):
            params = dict(kwargs)
            params["seed"] = i
            self.results.append(self._run_single(params))
        return self.results

    def run_parallel(
        self,
        num_runs: int,
        processes: int | None = None,
        **kwargs: Any
    ) -> list[Any]:
        """
        Execute Monte Carlo iterations across multiple processor cores.

        Parameters
        ----------
        num_runs : int
            Number of trials to execute.
        processes : Optional[int]
            Number of parallel workers. Defaults to machine CPU count.
        **kwargs
            Parameters for simulation configuration.

        Returns
        -------
        List[Any]
            Aggregated results.
        """
        self.results = []
        pool_kwargs = []
        for i in range(num_runs):
            params = dict(kwargs)
            params["seed"] = i
            pool_kwargs.append(params)

        # Attempt to use joblib for superior robustness in interactive environments (Jupyter)
        try:
            from joblib import Parallel, delayed
            self.results = Parallel(n_jobs=processes, backend="loky")(
                delayed(self._run_single)(p) for p in pool_kwargs
            )
        except ImportError:
            # Fallback to standard multiprocessing
            import multiprocessing as mp
            # Use 'spawn' or default context based on OS
            with mp.Pool(processes) as pool:
                self.results = pool.map(self._run_single, pool_kwargs)

        return self.results

    def get_analyzer(self) -> MonteCarloAnalyzer:
        """
        Produce a statistical analyzer for the current simulation results.
        
        Returns
        -------
        MonteCarloAnalyzer
            Instance of the analyzer initialized with current trial data.
        """
        return MonteCarloAnalyzer(self.results)
