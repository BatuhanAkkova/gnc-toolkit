import timeit
import numpy as np
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def run_benchmark(name, setup, stmt, number=10000):
    """Run a single benchmark item and print results."""
    t = timeit.Timer(stmt, setup=setup)
    try:
        # For Numba-accelerated functions, the first run includes compilation time.
        # We run it once during setup or before timing if needed.
        time_taken = t.timeit(number=number)
        avg_time = time_taken / number
        print(f"| {name:<35} | {number:<10} | {time_taken:<10.5f} | {avg_time * 1e6:<12.2f} |")
        return avg_time
    except Exception as e:
        print(f"| {name:<35} | ERROR: {str(e)[:20]}... | | |")
        return None

def main():
    print("# OpenGNC Benchmarks")
    
    print("\n## Propagators\n")
    print("| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |")
    print("|-------------------------------------|------------|----------------|---------------|")

    setup_prop = (
        "import numpy as np; "
        "from opengnc.propagators.kepler import KeplerPropagator; "
        "from opengnc.propagators.cowell import CowellPropagator; "
        "prop_kep = KeplerPropagator(); "
        "prop_cow = CowellPropagator(); "
        "r = np.array([7000e3, 0, 0]); v = np.array([0, 7.5e3, 0]); dt = 100.0"
    )
    run_benchmark("Kepler Propagator (Analytical)", setup_prop, "prop_kep.propagate(r, v, dt)", number=5000)
    run_benchmark("Cowell Propagator (RK4, dt=100, step=10)", setup_prop, "prop_cow.propagate(r, v, dt, dt_step=10.0)", number=500)

    print("\n## Gravity Models\n")
    print("| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |")
    print("|-------------------------------------|------------|----------------|---------------|")

    setup_grav = (
        "import numpy as np; "
        "from opengnc.disturbances.gravity import J2Gravity, HarmonicsGravity; "
        "grav_j2 = J2Gravity(); "
        "grav_harm = HarmonicsGravity(n_max=20, m_max=20); "
        "r = np.array([7000e3, 0, 0]); jd = 2460000.5; "
        "# Warm up Numba; "
        "grav_harm.get_acceleration(r, jd)"
    )
    run_benchmark("J2 Gravity Acceleration", setup_grav, "grav_j2.get_acceleration(r)", number=10000)
    run_benchmark("Harmonics Gravity (EGM2008 20x20)", setup_grav, "grav_harm.get_acceleration(r, jd)", number=1000)

    print("\n## Atmospheric Density\n")
    print("| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |")
    print("|-------------------------------------|------------|----------------|---------------|")

    setup_dens = (
        "import numpy as np; from datetime import datetime; "
        "from opengnc.environment.density import Exponential, HarrisPriester, NRLMSISE00; "
        "dens_exp = Exponential(); "
        "dens_hp = HarrisPriester(); "
        "dens_msis = NRLMSISE00(); "
        "r = np.array([7000e3, 0, 0]); jd = 2460000.5; dt_obj = datetime(2024, 1, 1)"
    )
    run_benchmark("Exponential Density", setup_dens, "dens_exp.get_density(r, jd)", number=10000)
    run_benchmark("Harris-Priester Density", setup_dens, "dens_hp.get_density(r, jd)", number=5000)
    run_benchmark("NRLMSISE-00 Density (via pymsis)", setup_dens, "dens_msis.get_density(r, dt_obj)", number=100)

    print("\n## Kalman Filters\n")
    print("| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |")
    print("|-------------------------------------|------------|----------------|---------------|")

    # Benchmark KF
    setup_kf = (
        "import numpy as np; "
        "from opengnc.kalman_filters.kf import KF; "
        "kf = KF(dim_x=6, dim_z=3); "
        "z = np.array([1.1, 2.2, 3.3])"
    )
    run_benchmark("KF Predict", setup_kf, "kf.predict()", number=10000)
    run_benchmark("KF Update", setup_kf, "kf.update(z)", number=10000)

    # Benchmark EKF
    setup_ekf = (
        "import numpy as np; "
        "from opengnc.kalman_filters.ekf import EKF; "
        "ekf = EKF(dim_x=6, dim_z=3); "
        "fx = lambda x, dt, u, **kwargs: x; "
        "fj = lambda x, dt, u, **kwargs: np.eye(6); "
        "hx = lambda x, **kwargs: x[:3]; "
        "hj = lambda x, **kwargs: np.eye(3, 6); "
        "z = np.array([1.1, 2.2, 3.3]); dt = 0.1"
    )
    run_benchmark("EKF Predict", setup_ekf, "ekf.predict(fx, fj, dt)", number=5000)
    run_benchmark("EKF Update", setup_ekf, "ekf.update(z, hx, hj)", number=5000)

    # Benchmark UKF
    setup_ukf = (
        "import numpy as np; "
        "from opengnc.kalman_filters.ukf import UKF; "
        "ukf = UKF(dim_x=6, dim_z=3); "
        "fx = lambda x, dt, **kwargs: x; "
        "hx = lambda x, **kwargs: x[:3]; "
        "z = np.array([1.1, 2.2, 3.3]); dt = 0.1"
    )
    run_benchmark("UKF Predict (dim=6)", setup_ukf, "ukf.predict(dt, fx)", number=1000)
    run_benchmark("UKF Update (dim=6)", setup_ukf, "ukf.update(z, hx)", number=1000)

    print("\n## Mission Design & Guidance\n")
    print("| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |")
    print("|-------------------------------------|------------|----------------|---------------|")

    setup_guidance = (
        "import numpy as np; "
        "from opengnc.guidance.maneuvers import hohmann_transfer, optimal_combined_maneuver; "
        "r1 = 7000.0; r2 = 42000.0; di = 0.5"
    )
    run_benchmark("Hohmann Transfer", setup_guidance, "hohmann_transfer(r1, r2)", number=10000)
    run_benchmark("Optimal Combined Maneuver", setup_guidance, "optimal_combined_maneuver(r1, r2, di)", number=100)

    print("\n## Coordinate Frames & Time\n")
    print("| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |")
    print("|-------------------------------------|------------|----------------|---------------|")

    setup_utils_complex = (
        "import numpy as np; "
        "from opengnc.utils.frame_conversion import eci2ecef, ecef2eci, eci2llh; "
        "from opengnc.utils.time_utils import calc_gmst, calc_jd; "
        "r = np.array([7000e3, 0, 0]); v = np.array([0, 7.5e3, 0]); jd = 2460000.5"
    )
    run_benchmark("ECI to ECEF Conversion", setup_utils_complex, "eci2ecef(r, v, jd)", number=5000)
    run_benchmark("ECI to LLH (Iterative)", setup_utils_complex, "eci2llh(r, jd)", number=2000)
    run_benchmark("GMST Calculation", setup_utils_complex, "calc_gmst(jd)", number=10000)

    print("\n## Classical Control & Math\n")
    print("| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |")
    print("|-------------------------------------|------------|----------------|---------------|")

    # Benchmark PID
    setup_pid = (
        "from opengnc.classical_control.pid import PID; "
        "pid = PID(kp=1.0, ki=0.1, kd=0.01); "
        "err = 0.5; dt = 0.1"
    )
    run_benchmark("PID Update", setup_pid, "pid.update(err, dt)", number=50000)

    # Benchmark quat math
    setup_quat_math = (
        "import numpy as np; "
        "from opengnc.utils.quat_utils import quat_mult, quat_conj, quat_normalize; "
        "q1 = np.array([0.1, 0.2, 0.3, 0.9]); q1 /= np.linalg.norm(q1); "
        "q2 = np.array([0.0, 0.0, 1.0, 0.0])"
    )
    run_benchmark("quat_mult", setup_quat_math, "quat_mult(q1, q2)")
    run_benchmark("quat_conj", setup_quat_math, "quat_conj(q1)")
    run_benchmark("quat_normalize", setup_quat_math, "quat_normalize(q1)")

    print("\n## Attitude Determination & Integrators\n")
    print("| Operation                           | Iterations | Total Time (s) | Avg Time (µs) |")
    print("|-------------------------------------|------------|----------------|---------------|")

    # Benchmark TRIAD
    setup_triad = (
        "import numpy as np; "
        "from opengnc.attitude_determination.triad import triad; "
        "b1 = np.array([1.0, 0.0, 0.0]); b2 = np.array([0.0, 1.0, 0.0]); "
        "r1 = np.array([1.0, 0.0, 0.0]); r2 = np.array([0.0, 1.0, 0.0])"
    )
    run_benchmark("TRIAD Determination", setup_triad, "triad(b1, b2, r1, r2)", number=20000)

    # Benchmark RK4
    setup_rk4 = (
        "import numpy as np; "
        "from opengnc.integrators.rk4 import RK4; "
        "f = lambda t, y, **kwargs: -0.1 * y; "
        "rk = RK4(); "
        "y = np.array([1.0, 2.0, 3.0]); "
        "dt = 0.1; t = 0.0"
    )
    run_benchmark("RK4 Step (3D Linear)", setup_rk4, "rk.step(f, t, y, dt)", number=50000)

if __name__ == "__main__":
    main()
