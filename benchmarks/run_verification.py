import numpy as np
import os
import sys
from opengnc.simulation.monte_carlo import MonteCarloSim
from opengnc.simulation.monte_carlo_analyzer import MonteCarloAnalyzer

# ---------------------------------------------------------
# 1. Define Stochastic Scenario for Verification
# ---------------------------------------------------------

class StochasticSimulator:
    def __init__(self, seed, tf=15.0, dt=0.05):
        self.rng = np.random.default_rng(seed)
        self.tf = tf
        self.dt = dt
        self.time = 0.0
        self.state = np.array([0.0, 0.0]) # [pos, vel]
        self.target = 10.0
        
        # Stochastic parameters
        self.process_noise_std = 0.02
        self.sensor_noise_std = 0.05
        
    def run(self):
        history = {"time": [], "pos": [], "vel": [], "error": [], "cov": []}
        
        # Consistent covariance for NIS demonstration
        # In a real system, this would come from the KF.P matrix
        p_est = np.array([[self.sensor_noise_std**2, 0.0], [0.0, 0.01]])
        
        while self.time < self.tf:
            # PID-like Control
            u = 3.0 * (self.target - self.state[0]) - 2.5 * self.state[1]
            
            # Dynamics
            accel = u + self.rng.normal(0, self.process_noise_std)
            self.state[1] += accel * self.dt
            self.state[0] += self.state[1] * self.dt
            
            # Stochastic Measurement
            meas = self.state[0] + self.rng.normal(0, self.sensor_noise_std)
            # The innovation/error we are testing for consistency
            error = np.array([meas - self.state[0], 0.0])
            
            history["time"].append(self.time)
            history["pos"].append(self.state[0])
            history["vel"].append(self.state[1])
            history["error"].append(error)
            history["cov"].append(p_est)
            
            self.time += self.dt
            
        for k in history:
            history[k] = np.array(history[k])
            
        return history

def simulator_factory(seed, **kwargs):
    return StochasticSimulator(seed, **kwargs)

# ---------------------------------------------------------
# 2. Main Verification Execution
# ---------------------------------------------------------

def main():
    print("="*60)
    print(" OPEN GNC PHASE 3 VERIFICATION SUITE ")
    print("="*60)
    
    num_runs = 100
    print(f"Running {num_runs} Monte Carlo trials...")
    
    mc = MonteCarloSim(simulator_factory)
    # Give it enough time to settle (15s)
    results = mc.run_parallel(num_runs=num_runs, tf=15.0, dt=0.05)
    
    analyzer = mc.get_analyzer()
    
    print("\n[PERFORMANCE MARGINS]")
    pos_stats = analyzer.get_aggregate_stats("pos")
    final_pos_mean = pos_stats["mean"][-1]
    # Calculate 3-sigma envelope at end of mission
    final_pos_3sig = 3 * pos_stats["std"][-1]
    
    print(f"Final Position Mean: {final_pos_mean:.4f} (Target: 10.0)")
    print(f"Final Position 3-Sigma Margin: +/- {final_pos_3sig:.4f}")
    
    print("\n[COVARIANCE CONSISTENCY]")
    consistency = analyzer.check_covariance_consistency("error", "cov")
    print(f"Average NIS: {consistency.get('avg_nis', 0):.4f}")
    print(f"Expected (State Dim): {consistency.get('expected', 0)}")
    print(f"Consistency Ratio: {consistency.get('consistency_ratio', 0):.4f}")
    
    print("\n[RELIABILITY ANALYSIS]")
    # Failure if final position error > 0.1 (tighter bound)
    failure_criteria = lambda res: abs(res["pos"][-1] - 10.0) > 0.1
    summary = analyzer.summarize_failures(failure_criteria)
    print(f"Failure Rate: {summary['failure_rate']*100:.1f}%")
    print(f"Reliability: {summary['reliability']*100:.1f}%")
    
    print("\n" + "="*60)
    print(" VERIFICATION COMPLETE ")
    print("="*60)

if __name__ == "__main__":
    main()
