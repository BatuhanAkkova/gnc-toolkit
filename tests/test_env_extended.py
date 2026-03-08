import pytest
import numpy as np
import datetime
from gnc_toolkit.environment.mag_field import wmm_field
from gnc_toolkit.environment.space_weather import SpaceWeather
from gnc_toolkit.environment.radiation import RadiationModel
from gnc_toolkit.environment.thermal import ThermalEnvironment

def test_wmm_field_mock(mocker):
    # Mock ppigrf.igrf since wmm_field uses it as a proxy
    mock_igrf = mocker.patch("ppigrf.igrf")
    mock_igrf.return_value = [1.1e-5, 2.1e-5, 3.1e-5]
    
    lat, lon, alt = 45.0, -120.0, 500.0
    date = datetime.datetime(2025, 1, 1)
    
    res = wmm_field(lat, lon, alt, date)
    assert len(res) == 3
    assert np.allclose(res, [1.1e-5, 2.1e-5, 3.1e-5])

def test_space_weather():
    sw = SpaceWeather(f107=140.0, ap=20.0)
    indices = sw.get_indices()
    assert indices['f107'] == 140.0
    assert indices['ap'] == 20.0
    assert indices['kp'] > 0
    
    sw.set_solar_flux(180.0)
    assert sw.f107 == 180.0
    assert sw.f107_avg == 180.0

def test_radiation_model():
    rad = RadiationModel()
    tid = rad.estimate_tid(500, 51.6, 365)
    assert tid > 0
    
    seu_rate = rad.estimate_seu_rate(500)
    assert seu_rate > 0

def test_thermal_environment():
    thermal = ThermalEnvironment(albedo_coeff=0.3)
    
    # Test Solar Flux
    solar = thermal.get_solar_flux(1.0)
    assert np.isclose(solar, 1361.0)
    
    # Test Albedo (Nadir facing Sat-Sun aligned)
    r_sat = np.array([7000, 0, 0])
    r_sun = np.array([149e6, 0, 0])
    albedo = thermal.get_albedo_flux(r_sat, r_sun)
    assert albedo > 0
    assert albedo < 1361.0 * 0.35 # Should be around 1361 * 0.3
    
    # Night side Albedo
    r_sun_night = np.array([-149e6, 0, 0])
    albedo_night = thermal.get_albedo_flux(r_sat, r_sun_night)
    assert albedo_night == 0.0

    # Earth IR
    ir = thermal.get_earth_ir_flux()
    assert ir == 230.0
