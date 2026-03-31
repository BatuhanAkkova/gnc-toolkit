import numpy as np

def fast_simulator(seed, **kwargs):
    """
    A fast mock simulator for demonstration.
    Designed to be importable for robust multiprocessing on Windows.
    """
    np.random.seed(seed)
    
    # Simulation duration and resolution
    tf = kwargs.get("tf", 10.0)
    dt = kwargs.get("dt", 0.1)
    n_steps = int(tf / dt)
    
    t = np.linspace(0, tf, n_steps)
    
    # Generate position error with drift and noise
    # Simple model: error = 0.1 * t + noise
    pos_error = 0.1 * t + np.random.normal(0, 0.05, n_steps)
    
    return {"time": t, "pos_error": pos_error}
