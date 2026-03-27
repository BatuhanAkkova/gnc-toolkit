import timeit
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Try to import, handle error if not found (though it should be there)
try:
    from opengnc.utils.state_conversion import quat_to_dcm, euler_to_dcm
except ImportError:
    print("Error: Could not import opengnc. Ensure it is installed or in PYTHONPATH.")
    sys.exit(1)

def run_benchmark(name, setup, stmt, number=10000):
    """Run a single benchmark item and print results."""
    t = timeit.Timer(stmt, setup=setup)
    try:
        time_taken = t.timeit(number=number)
        avg_time = time_taken / number
        print(f"| {name:<25} | {number:<10} | {time_taken:<10.5f} | {avg_time * 1e6:<12.2f} |")
        return avg_time
    except Exception as e:
        print(f"| {name:<25} | ERROR: {str(e)[:20]}... | | |")
        return None

def main():
    print("# OpenGNC Benchmarks")
    print("\n## Attitude Conversions\n")
    print("| Operation                 | Iterations | Total Time (s) | Avg Time (µs) |")
    print("|---------------------------|------------|----------------|---------------|")

    # Benchmark quat_to_dcm
    setup_quat = (
        "import numpy as np; "
        "from opengnc.utils.state_conversion import quat_to_dcm; "
        "q = np.array([0.0, 0.0, 0.0, 1.0])"
    )
    run_benchmark("quat_to_dcm", setup_quat, "quat_to_dcm(q)")

    # Benchmark euler_to_dcm
    setup_euler = (
        "import numpy as np; "
        "from opengnc.utils.state_conversion import euler_to_dcm; "
        "angle = np.array([0.1, 0.2, 0.3]); "
        "seq = '123'"
    )
    run_benchmark("euler_to_dcm (123)", setup_euler, "euler_to_dcm(angle, seq)")

if __name__ == "__main__":
    main()




