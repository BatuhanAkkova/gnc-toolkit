# GNC Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A comprehensive Guidance, Navigation, and Control (GNC) toolkit for spacecraft simulation, mission analysis, and state estimation. Built with Python, this toolkit provides high-fidelity environment models, disturbance calculations, and advanced control/estimation algorithms.

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
- **PyMSIS**: NRLMSISE-00 atmospheric density model.
- **PPIGRF**: IGRF-13 geomagnetic field model.

### Quick Start Guide

The package is structured for easy access to its submodules. Here are some common use cases:

```python
# Import estimation filters
from gnc_toolkit.kalman_filters.mekf import MEKF
from gnc_toolkit.kalman_filters.ukf import UKF_Attitude

# Import high-fidelity environment models
from gnc_toolkit.environment.density import NRLMSISE00, HarrisPriester
from gnc_toolkit.environment.mag_field import igrf_field

# Calculate disturbances
from gnc_toolkit.disturbances.gravity import HarmonicsGravity
from gnc_toolkit.disturbances.drag import LumpedDrag

# Attitude utilities
from gnc_toolkit.utils.quat_utils import quat_rot, quat_mult
```

## Project Overview

The **GNC Toolkit** is designed to support the full lifecycle of small satellite missions, with a particular focus on Very Low Earth Orbit (VLEO) environments and complex attitude control scenarios.

- **High-Fidelity Environment**: Accurate models for atmospheric density (NRLMSISE-00), geomagnetic fields (IGRF-13), and solar ephemeris.
- **Physical Disturbances**: Modeling of J2 and EGM2008 Spherical Harmonics, atmospheric drag with co-rotating atmosphere, and Solar Radiation Pressure (SRP).
- **Advanced State Estimation**: Multiplicative Extended Kalman Filter (MEKF) for attitude, and various filters (EKF, UKF) for orbit and state determination.
- **Optimal & Nonlinear Control**: Support for LQR, MPC (Linear/Nonlinear), Sliding Mode Control, and B-dot detumbling.

## Core Features

### Environment & Physical Models
- **Gravity**: Two-body, J2, and EGM2008 Spherical Harmonics (recursive implementation).
- **Atmosphere**: Exponential, Harris-Priester (diurnal bulge), and NRLMSISE-00.
- **Magnetic Field**: Tilted Dipole and IGRF-13.
- **Solar**: Analytical solar position and shadow models (umbra/penumbra).

### Estimation & Navigation
- **Kalman Filtering**: KF, EKF, MEKF (for quaternions), and UKF.
- **Attitude Determination**: Deterministic TRIAD and QUEST algorithms.
- **Sensors**: Realistic Star Tracker, Sun Sensor, Magnetometer, and Gyroscope models with bias and noise.

### Guidance & Mission Analysis
- **Orbital Maneuvers**: Hohmann, Bi-elliptic, Phasing, and Inclination changes.
- **Rendezvous**: Lambert Solver (Universal Variables), Clohessy-Wiltshire (CW) equations, and CW targeting.
- **Propagators**: High-order Keplerian and Cowell numerical propagators (RK4, RK45, DOP853).

### Control Systems
- **Classical**: PID controllers and B-dot detumbling logic.
- **Optimal**: LQR (Algebraic Riccati Equation solver) and LQE.
- **Robust/Modern**: Sliding Mode Control and Model Predictive Control (MPC).
- **Actuators**: Reaction Wheels (momentum management) and Thrusters (Chemical/Electric).

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

**Batuhan Akkova**
[Email](mailto:batuhanakkova1@gmail.com)
