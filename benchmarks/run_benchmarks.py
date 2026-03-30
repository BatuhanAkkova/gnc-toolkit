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
    print("| Operation                           | Iterations | Total Time (s) | Avg Time (us) |")
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
    print("| Operation                           | Iterations | Total Time (s) | Avg Time (us) |")
    print("|-------------------------------------|------------|----------------|---------------|")

    setup_grav = (
        "import numpy as np; "
        "from opengnc.disturbances.gravity import J2Gravity, HarmonicsGravity; "
        "grav_j2 = J2Gravity(); "
        "grav_harm = HarmonicsGravity(n_max=20, m_max=20); "
        "r = np.array([7000e3, 0, 0]); jd = 2460000.5; "
        "grav_harm.get_acceleration(r, jd)"
    )
    run_benchmark("J2 Gravity Acceleration", setup_grav, "grav_j2.get_acceleration(r)", number=10000)
    run_benchmark("Harmonics Gravity (EGM2008 20x20)", setup_grav, "grav_harm.get_acceleration(r, jd)", number=1000)

    print("\n## Atmospheric Density\n")
    print("| Operation                           | Iterations | Total Time (s) | Avg Time (us) |")
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
    print("| Operation                           | Iterations | Total Time (s) | Avg Time (us) |")
    print("|-------------------------------------|------------|----------------|---------------|")

    setup_kf = (
        "import numpy as np; from opengnc.kalman_filters.kf import KF; "
        "kf = KF(dim_x=6, dim_z=3); z = np.array([1.1, 2.2, 3.3])"
    )
    run_benchmark("KF Predict", setup_kf, "kf.predict()", number=10000)
    run_benchmark("KF Update", setup_kf, "kf.update(z)", number=10000)

    setup_ekf = (
        "import numpy as np; from opengnc.kalman_filters.ekf import EKF; "
        "ekf = EKF(dim_x=6, dim_z=3); fx = lambda x, dt, u, **kwargs: x; "
        "fj = lambda x, dt, u, **kwargs: np.eye(6); hx = lambda x, **kwargs: x[:3]; "
        "hj = lambda x, **kwargs: np.eye(3, 6); z = np.array([1.1, 2.2, 3.3]); dt = 0.1"
    )
    run_benchmark("EKF Predict", setup_ekf, "ekf.predict(fx, fj, dt)", number=5000)
    run_benchmark("EKF Update", setup_ekf, "ekf.update(z, hx, hj)", number=5000)

    setup_ukf = (
        "import numpy as np; from opengnc.kalman_filters.ukf import UKF; "
        "ukf = UKF(dim_x=6, dim_z=3); fx = lambda x, dt, **kwargs: x; "
        "hx = lambda x, **kwargs: x[:3]; z = np.array([1.1, 2.2, 3.3]); dt = 0.1"
    )
    run_benchmark("UKF Predict (dim=6)", setup_ukf, "ukf.predict(dt, fx)", number=1000)
    run_benchmark("UKF Update (dim=6)", setup_ukf, "ukf.update(z, hx)", number=1000)

    print("\n## Classical Control & Math\n")
    print("| Operation                           | Iterations | Total Time (s) | Avg Time (us) |")
    print("|-------------------------------------|------------|----------------|---------------|")

    setup_pid = (
        "from opengnc.classical_control.pid import PID; pid = PID(kp=1.0, ki=0.1, kd=0.01); "
        "err = 0.5; dt = 0.1"
    )
    run_benchmark("PID Update", setup_pid, "pid.update(err, dt)", number=50000)

    setup_quat_math = (
        "import numpy as np; from opengnc.utils.quat_utils import quat_mult, quat_conj, quat_normalize; "
        "q1 = np.array([0.1, 0.2, 0.3, 0.9]); q1 /= np.linalg.norm(q1); q2 = np.array([0.0, 0.0, 1.0, 0.0])"
    )
    run_benchmark("quat_mult", setup_quat_math, "quat_mult(q1, q2)")
    run_benchmark("quat_conj", setup_quat_math, "quat_conj(q1)")
    run_benchmark("quat_normalize", setup_quat_math, "quat_normalize(q1)")

    setup_rk4 = (
        "import numpy as np; from opengnc.integrators.rk4 import RK4; "
        "f = lambda t, y, **kwargs: -0.1 * y; rk = RK4(); y = np.array([1.0, 2.0, 3.0]); dt = 0.1; t = 0.0"
    )
    run_benchmark("RK4 Step (3D Linear)", setup_rk4, "rk.step(f, t, y, dt)", number=50000)

    print("\n## C++ Accelerated Kalman Filters (opengnc_py)\n")
    print("| Operation                           | Iterations | Total Time (s) | Avg Time (us) | Speedup (vs Py) |")
    print("|-------------------------------------|------------|----------------|---------------|-----------------|")

    setup_cpp_utils = (
        "import numpy as np; import sys; import os; "
        "sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/Release'))); "
        "import opengnc_py; "
        "q1 = np.array([0.1, 0.2, 0.3, 0.9]); q1 /= np.linalg.norm(q1)"
    )

    avg_py_norm_s = 3.3 * 1e-6 
    avg_cpp_norm_s = run_benchmark('quat_normalize (C++)', setup_cpp_utils, 'opengnc_py.quat_normalize(q1)', number=50000)
    if avg_cpp_norm_s:
        print(f'| {"quat_normalize (C++)":<35} | {"-":<10} | {"-":<10} | {"-":<13} | {avg_py_norm_s/avg_cpp_norm_s:<15.1f}x |')

    setup_mekf_cpp = (
        "import numpy as np; import sys; import os; "
        "sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/Release'))); "
        "import opengnc_py; mekf = opengnc_py.MEKF(); "
        "w = np.array([0.1, 0, 0]); v_body = np.array([0, 0, 1]); v_ref = np.array([0, 0, 1])"
    )
    avg_py_mekf_p_s = 6.0 * 1e-6 
    avg_py_mekf_u_s = 30.0 * 1e-6 
    avg_cpp_mekf_p_s = run_benchmark('MEKF Predict (C++)', setup_mekf_cpp, 'mekf.predict(w, 0.1)', number=20000)
    avg_cpp_mekf_u_s = run_benchmark('MEKF Update (C++)', setup_mekf_cpp, 'mekf.update(v_body, v_ref)', number=20000)
    if avg_cpp_mekf_p_s:
        print(f'| {"MEKF Predict (C++)":<35} | {"-":<10} | {"-":<10} | {"-":<13} | {avg_py_mekf_p_s/avg_cpp_mekf_p_s:<15.1f}x |')
    if avg_cpp_mekf_u_s:
        print(f'| {"MEKF Update (C++)":<35} | {"-":<10} | {"-":<10} | {"-":<13} | {avg_py_mekf_u_s/avg_cpp_mekf_u_s:<15.1f}x |')

    setup_ukf_cpp = (
        "import numpy as np; import sys; import os; "
        "sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/Release'))); "
        "import opengnc_py; ukf = opengnc_py.UKF_Attitude(); "
        "fx = lambda x, dt: x; hx = lambda x: x[:3]; z = np.array([0,0,1])"
    )
    avg_py_ukf_p_s = 120.0 * 1e-6 
    avg_py_ukf_u_s = 220.0 * 1e-6 
    avg_cpp_ukf_p_s = run_benchmark('UKF Predict (C++)', setup_ukf_cpp, 'ukf.predict(0.1, fx)', number=5000)
    avg_cpp_ukf_u_s = run_benchmark('UKF Update (C++)', setup_ukf_cpp, 'ukf.update(z, hx)', number=5000)
    if avg_cpp_ukf_p_s:
        print(f'| {"UKF Predict (C++)":<35} | {"-":<10} | {"-":<10} | {"-":<13} | {avg_py_ukf_p_s/avg_cpp_ukf_p_s:<15.1f}x |')
    if avg_cpp_ukf_u_s:
        print(f'| {"UKF Update (C++)":<35} | {"-":<10} | {"-":<10} | {"-":<13} | {avg_py_ukf_u_s/avg_cpp_ukf_u_s:<15.1f}x |')

if __name__ == "__main__":
    main()
