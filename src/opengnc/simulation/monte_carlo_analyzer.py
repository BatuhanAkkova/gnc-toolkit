import numpy as np
from typing import Any, Dict, List

class MonteCarloAnalyzer:
    """
    Statistical analysis suite for Monte Carlo simulations.
    
    Provides specialized tools for calculating performance margins, 
    stability metrics, and covariance consistency proofs.
    """

    def __init__(self, results: List[Dict[str, Any]]) -> None:
        """
        Initialize with a list of simulation result dictionaries.
        
        Each dictionary should contain time-series data or terminal states.
        """
        self.results = results
        if not results:
            raise ValueError("No results provided to MonteCarloAnalyzer.")

    def get_aggregate_stats(self, key: str) -> Dict[str, np.ndarray]:
        """
        Extract mean, std dev, and 3-sigma bounds for a specific telemetry key.
        
        Assumes the key points to a 1D or 2D numpy array in each result.
        """
        data = [res[key] for res in self.results if key in res]
        if not data:
            return {}

        arr = np.array(data) # Shape: (NumRuns, NumSteps, Dim) or (NumRuns, Dim)
        
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        
        return {
            "mean": mean,
            "std": std,
            "sigma_3_upper": mean + 3 * std,
            "sigma_3_lower": mean - 3 * std,
            "min": np.min(arr, axis=0),
            "max": np.max(arr, axis=0),
            "median": np.median(arr, axis=0)
        }

    def check_covariance_consistency(self, error_key: str, cov_key: str) -> Dict[str, float]:
        """
        Perform a consistency test (e.g., NIS or NEES).
        
        Calculates the average quadratic form: e^T * P^-1 * e
        Should be close to the dimension of the state if P is consistent.
        """
        errors = [res[error_key] for res in self.results if error_key in res]
        covs = [res[cov_key] for res in self.results if cov_key in res]
        
        if not errors or not covs:
            return {"status": "missing_data"}

        nis_values = []
        for e, p in zip(errors, covs):
            # Assumes e and p are time-series or terminal
            # If time-series, we take the average over time
            if e.ndim > 1:
                run_nis = []
                for i in range(len(e)):
                    try:
                        inv_p = np.linalg.inv(p[i])
                        run_nis.append(e[i].T @ inv_p @ e[i])
                    except np.linalg.LinAlgError:
                        continue
                nis_values.append(np.mean(run_nis))
            else:
                try:
                    inv_p = np.linalg.inv(p)
                    nis_values.append(e.T @ inv_p @ e)
                except np.linalg.LinAlgError:
                    continue

        avg_nis = np.mean(nis_values)
        return {
            "avg_nis": avg_nis,
            "expected": float(errors[0].shape[-1]),
            "consistency_ratio": avg_nis / errors[0].shape[-1]
        }

    def summarize_failures(self, criteria_func: callable) -> Dict[str, Any]:
        """
        Calculate failure rates based on a user-defined criteria function.
        """
        failures = 0
        for res in self.results:
            if criteria_func(res):
                failures += 1
        
        rate = failures / len(self.results)
        return {
            "total_runs": len(self.results),
            "failures": failures,
            "failure_rate": rate,
            "reliability": 1.0 - rate
        }
