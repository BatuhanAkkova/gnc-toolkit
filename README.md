# GNC Toolkit

A comprehensive Guidance, Navigation, and Control (GNC) toolkit for satellite simulation and estimation. This project provides high-fidelity environment models, disturbance calculations, and advanced estimation algorithms in both Python and C++.

## Getting Started

### Installation

To use this toolkit in your own projects, you can install it in editable mode:

```bash
git clone https://github.com/BatuhanAkkova/gnc_toolkit.git
cd gnc_toolkit
pip install -e .
```

### Dependencies

The following packages are required:
- **NumPy**: Linear algebra and array operations.
- **SciPy**: Numerical integration and optimization.
- **PyMSIS**: NRLMSISE-00 atmospheric model.
- **PPIGRF**: IGRF-13 magnetic field model.

### Importing Guide

The package is structured for easy access to its submodules. Here are some common examples:

```python
# Import estimation filters
from gnc_toolkit.kalman_filters.mekf import MEKF
from gnc_toolkit.kalman_filters.ukf import UKF_Attitude

# Import environment models
from gnc_toolkit.environment.density import NRLMSISE00
from gnc_toolkit.environment.mag_field import igrf_field

# Import disturbances
from gnc_toolkit.disturbances.gravity import HarmonicsGravity
from gnc_toolkit.disturbances.drag import LumpedDrag

# Import utilities
from gnc_toolkit.utils.quat_utils import quat_rot, quat_mult
```

## Project Overview

The **GNC Toolkit** is designed to support mission analysis, simulation, and flight software development for small satellites, with a particular emphasis on Very Low Earth Orbit (VLEO) environments. It includes modular components for:

- **Environment Modeling**: Accurate calculations for atmospheric density, magnetic fields, and solar positions.
- **Disturbance Analysis**: High-fidelity models for gravity (including EGM2008 spherical harmonics), atmospheric drag, and solar radiation pressure.
- **State Estimation**: Robust filtering algorithms including EKF, MEKF, and UKF for orbit and attitude determination.
- **Cross-Platform Implementation**: Core logic implemented in Python for rapid prototyping and C++ for performance-critical applications.

## Features

### Environment & Disturbances
- **Gravity Models**: Two-body, J2, and high-fidelity EGM2008 Spherical Harmonics.
- **Atmospheric Drag**: Lumped drag models with Harris-Priester density estimation.
- **Solar Radiation Pressure**: Cannonball model with shadow effects and solar position tracking.
- **Magnetic Field**: Geomagnetic field modeling for sensor simulation.

### Estimation (Kalman Filters)
- **KF**: Standard Kalman Filter for linear systems.
- **EKF**: Extended Kalman Filter for non-linear state estimation.
- **MEKF**: Multiplicative Extended Kalman Filter for attitude estimation using quaternions.
- **UKF**: Unscented Kalman Filter for handling high degrees of non-linearity.

### Numerical Integrators
- **RK4**: Fixed step Runge-Kutta 4.
- **RK45**: Adaptive step Runge-Kutta-Fehlberg.
- **RK853**: High-order adaptive Dormand-Prince.

### Attitude Determination
- **TRIAD**: Deterministic method using two vectors.
- **QUEST**: Deterministic method using multiple vectors.

### Sensors & Actuators
- **Sensors**: Base framework with implementations for:
    - **Star Tracker**: Quaternion output with noise and bias models.
    - **Sun Sensor**: Vector measurement with field-of-view and noise.
    - **Magnetometer**: Magnetic field vector measurement with bias and scaling.
    - **Gyroscope**: Angular rate measurement with bias instability (Random Walk).
- **Actuators**: Base framework with implementations for:
    - **Reaction Wheels**: Torque command with speed/momentum saturation.
    - **Magnetorquers**: Dipole command with saturation.
    - **Thrusters**:
        - **Chemical**: PWM logic for analyzing on-time vs average thrust, minimum impulse bit enforcement.
        - **Electric**: Power consumption modeling based on efficiency and Isp.

### Propagators
- **Two-Body**: Two-body orbit propagation.
- **Cowell**: Numerical integration of equations of motion with disturbances.

### Attitude Dynamics
- **Rigid Body**: Euler equations for rigid body motion.

### Guidance & Mission Analysis Tools
- **Orbital Maneuvers**: Hohmann transfer, bi-elliptic transfer, phasing manouvers, plane change, combined plane change.
- **Rendezvous**: Lambert problem, CW equations, CW targeting.

### Classical Control Algorithms
- **PID**: Proportional-Integral-Derivative controller.
- **B-dot**: Magnetic detumbling controller.

## TODO
### Optimal Control Algorithms
- [ ] LQR controller (infinite horizon, via solving ARE)
- [ ] LQE controller (duality of LQR for estimation)
- [ ] Sliding mode controller
- [ ] MPC
- [ ] Feedback Linearization (cancel out non-linear terms)

### C++ Library
- [ ] Develop cpp library further
