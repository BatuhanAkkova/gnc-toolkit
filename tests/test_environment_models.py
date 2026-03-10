import numpy as np
import pytest
from datetime import datetime
from gnc_toolkit.environment.density import JB2008, CIRA72
from gnc_toolkit.environment.wind import AtmosphereCoRotation
from gnc_toolkit.environment.moon import Moon
from gnc_toolkit.utils.time_utils import calc_jd

def test_jb2008_density():
    model = JB2008()
    r_eci = np.array([7000.0, 0.0, 0.0]) * 1000.0 # 622 km altitude
    jd = 2451545.0
    rho = model.get_density(r_eci, jd)
    assert rho > 0
    assert rho < 1e-6

def test_cira72_density():
    model = CIRA72()
    r_eci = np.array([7000.0, 0.0, 0.0]) * 1000.0
    jd = 2451545.0
    rho1 = model.get_density(r_eci, jd)
    
    r_eci2 = np.array([8000.0, 0.0, 0.0]) * 1000.0
    rho2 = model.get_density(r_eci2, jd)
    
    assert rho1 > rho2 # Density should decrease with altitude
    assert rho1 > 0

def test_atmosphere_corotation():
    model = AtmosphereCoRotation()
    r_eci = np.array([7000.0, 0.0, 0.0]) * 1000.0 # Equator
    jd = 2451545.0
    v_wind = model.get_wind_velocity(r_eci, jd)
    
    # At equator (7000km), v = omega * r
    # v ~ 7.29e-5 * 7e6 ~ 510 m/s
    assert np.allclose(v_wind, [0, 510.448, 0], atol=1.0)
    
    r_pole = np.array([0, 0, 7000.0]) * 1000.0
    v_wind_pole = model.get_wind_velocity(r_pole, jd)
    assert np.allclose(v_wind_pole, [0, 0, 0])

def test_moon_position():
    model = Moon()
    jd = 2451545.0 # J2000
    r_moon = model.calculate_moon_eci(jd)
    
    # Moon distance is approx 384,400 km
    dist = np.linalg.norm(r_moon) / 1000.0
    assert dist > 350000 and dist < 410000
