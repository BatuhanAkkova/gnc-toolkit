Monte Carlo Verification
========================

This tutorial demonstrates how to use the Monte Carlo Simulation harness and the statistical analyzer for GNC system verification.

Overview
--------

For mission-critical spacecraft software, a single simulation run is insufficient to prove robustness. 
**OpenGNC** provides a high-performance Monte Carlo suite to analyze performance under stochastic variations in environment, sensor noise, and process disturbances.

Running a Monte Carlo Simulation
--------------------------------

The ``MonteCarloSim`` class facilitates parallel execution of multiple trials.

.. code-block:: python

    from opengnc.simulation.monte_carlo import MonteCarloSim
    
    def my_simulator(seed, **kwargs):
        # Initialize your MissionSimulator with the given seed
        sim = MySimulator(seed=seed, **kwargs)
        return sim.run()

    mc = MonteCarloSim(my_simulator)
    # Run 100 trials in parallel
    results = mc.run_parallel(num_runs=100, tf=100.0, dt=0.1)

Statistical Analysis & Verification
-----------------------------------

Once the trials are complete, you can use the ``MonteCarloAnalyzer`` to calculate performance margins.

.. code-block:: python

    analyzer = mc.get_analyzer()

    # 1. Calculate 3-Sigma Bounds
    pos_stats = analyzer.get_aggregate_stats("pos_error")
    print(f"Final Mean Error: {pos_stats['mean'][-1]}")
    print(f"3-Sigma Bound: {pos_stats['sigma_3_upper'][-1]}")

    # 2. Covariance Consistency Check (NIS)
    consistency = analyzer.check_covariance_consistency("error", "cov")
    print(f"Consistency Ratio: {consistency['consistency_ratio']}")

    # 3. Reliability Analysis
    # Define a failure criteria (e.g. error > 1 meter)
    failure_func = lambda res: abs(res["pos_error"][-1]) > 1.0
    summary = analyzer.summarize_failures(failure_func)
    print(f"Reliability Score: {summary['reliability'] * 100}%")

Verification Suite Tool
-----------------------

A pre-configured verification suite is available at ``benchmarks/run_verification.py`` for standard proof-of-mission analysis.

.. code-block:: bash

    $env:PYTHONPATH="src"; python benchmarks/run_verification.py
