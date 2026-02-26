import pytest
import numpy as np
import datetime
from gnc_toolkit.environment.density import Exponential, HarrisPriester, NRLMSISE00
from gnc_toolkit.environment.mag_field import tilted_dipole_field, igrf_field

def test_exponential_model():
    model = Exponential()
    r_eci = np.array([6500e3, 0.0, 0.0])
    jd = 2451545.0
    rho = model.get_density(r_eci, jd)
    assert rho > 0.0
    assert rho < 1.225
    
    r_space = np.array([50000e3, 0.0, 0.0])
    rho_space = model.get_density(r_space, jd)
    assert rho_space < rho
    assert rho_space >= 0.0

def test_harris_priester_model():
    model = HarrisPriester()
    r_eci = np.array([7000e3, 0.0, 0.0])
    jd = 2451545.0
    rho = model.get_density(r_eci, jd)
    assert rho > 0.0
    assert rho < 1.225
    
    r_space = np.array([50000e3, 0.0, 0.0])
    rho_space = model.get_density(r_space, jd)
    assert rho_space < rho
    assert rho_space >= 0.0

def test_nrlmsise_model(mocker):
    # Mock pymsis.calculate
    mock_calc = mocker.patch("pymsis.calculate")
    mock_calc.return_value = np.array([1.5e-12]) # Mock density
    
    model = NRLMSISE00()
    r_eci = np.array([7000e3, 0.0, 0.0]) 
    date = datetime.datetime(2024, 1, 1, 12, 0, 0)
    
    rho = model.get_density(r_eci, date)
    assert isinstance(rho, float)
    assert rho == 1.5e-12
    mock_calc.assert_called_once()

def test_tilted_dipole_field():
    # Test at some random position
    r_ecef = np.array([7000000.0, 0, 0])
    B = tilted_dipole_field(r_ecef)
    
    # Check shape and non-zero
    assert B.shape == (3,)
    assert np.linalg.norm(B) > 0.0
    
    # Check magnitude roughly (at 7000km, should be in microTesla range)
    # B approx: 3.12e-5 * (6371/7000)^3 = 2e-5 T
    assert np.linalg.norm(B) < 1e-4
    assert np.linalg.norm(B) > 1e-6

def test_igrf_field_mock(mocker):    
    # Mock ppigrf.igrf
    mock_igrf = mocker.patch("ppigrf.igrf")
    mock_igrf.return_value = [1e-5, 2e-5, 3e-5]
    
    lat = 0.0
    lon = 0.0
    alt = 600.0 # km
    date = datetime.datetime(2024, 1, 1)
    
    res = igrf_field(lat, lon, alt, date)
    assert len(res) == 3
    assert np.allclose(res, [1e-5, 2e-5, 3e-5])
    mock_igrf.assert_called_once()
