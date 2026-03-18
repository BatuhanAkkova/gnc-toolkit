import pytest
import numpy as np
from gnc_toolkit.mission_design.communications import (
    calculate_friis_link_budget,
    calculate_doppler_shift,
    calculate_atmospheric_attenuation
)

def test_friis_link_budget():
    p_tx_w = 10.0  # 10 W -> 10 dBW
    g_tx_db = 10.0
    g_rx_db = 15.0
    frequency_hz = 2.0e9  # 2 GHz (S-band)
    distance_m = 1000.0e3  # 1000 km
    
    # Lfs = 20*log10(1000e3) + 20*log10(2.0e9) - 147.554
    # = 120 + 186.02 - 147.554 = 158.466
    
    result = calculate_friis_link_budget(
        p_tx_w=p_tx_w,
        g_tx_db=g_tx_db,
        g_rx_db=g_rx_db,
        frequency_hz=frequency_hz,
        distance_m=distance_m
    )
    
    assert 'p_rx_dbw' in result
    assert 'p_rx_w' in result
    assert 'l_fs_db' in result
    
    # Check expected values
    # Lfs should be approx 158.466
    assert result['l_fs_db'] == pytest.approx(158.466, abs=0.1)
    
    # Prx (dBW) = 10 + 10 + 15 - 158.466 = -123.466
    assert result['p_rx_dbw'] == pytest.approx(-123.466, abs=0.1)
    
    # Include losses
    result_losses = calculate_friis_link_budget(
        p_tx_w=p_tx_w,
        g_tx_db=g_tx_db,
        g_rx_db=g_rx_db,
        frequency_hz=frequency_hz,
        distance_m=distance_m,
        losses_misc_db=2.0,
        l_atm_db=1.0
    )
    
    # Prx (dBW) = -123.466 - 2 - 1 = -126.466
    assert result_losses['p_rx_dbw'] == pytest.approx(-126.466, abs=0.1)

def test_doppler_shift():
    f_tx_hz = 2.0e9  # 2 GHz
    
    # Stationary Tx at origin
    r_tx = np.array([0.0, 0.0, 0.0])
    v_tx = np.array([0.0, 0.0, 0.0])
    
    # Rx moving away along X axis at 100 m/s
    r_rx = np.array([1000.0, 0.0, 0.0])
    v_rx = np.array([100.0, 0.0, 0.0])
    
    result = calculate_doppler_shift(f_tx_hz, r_rx, v_rx, r_tx, v_tx)
    
    # Expected shift: - f_tx * (v_rel / c)
    # v_rel = 100 m/s
    # c = 299792458
    # shift = - 2e9 * 100 / 299792458 = -667.128 Hz
    
    assert result['doppler_shift_hz'] == pytest.approx(-667.128, abs=0.001)
    assert result['f_rx_hz'] == pytest.approx(f_tx_hz - 667.128, abs=0.001)
    
    # Rx moving closer
    v_rx_closer = np.array([-100.0, 0.0, 0.0])
    result_closer = calculate_doppler_shift(f_tx_hz, r_rx, v_rx_closer, r_tx, v_tx)
    assert result_closer['doppler_shift_hz'] == pytest.approx(667.128, abs=0.001)

def test_atmospheric_attenuation():
    # S-band, Elevation 30 deg
    # Latm = 0.03 / sin(30) = 0.03 / 0.5 = 0.06 dB
    elevation = 30.0
    freq_s = 2.0e9
    
    l_atm_s = calculate_atmospheric_attenuation(elevation, freq_s)
    assert l_atm_s == pytest.approx(0.06)
    
    # Ka-band, Elevation 45 deg
    # Azentih = 0.35
    # Latm = 0.35 / sin(45) = 0.35 / 0.707106 = 0.49497 dB
    freq_ka = 20.0e9
    l_atm_ka = calculate_atmospheric_attenuation(45.0, freq_ka)
    assert l_atm_ka == pytest.approx(0.49497, abs=0.001)
    
    # Low elevation cap (should cap at 5 deg)
    l_atm_low = calculate_atmospheric_attenuation(1.0, freq_s)
    l_atm_cap = calculate_atmospheric_attenuation(5.0, freq_s)
    assert l_atm_low == l_atm_cap

def test_invalid_inputs():
    with pytest.raises(ValueError):
        calculate_friis_link_budget(-1, 10, 10, 1e9, 1000)
    with pytest.raises(ValueError):
        calculate_friis_link_budget(10, 10, 10, -1, 1000)
    with pytest.raises(ValueError):
        calculate_friis_link_budget(10, 10, 10, 1e9, -1)
        
    # Doppler shape error
    with pytest.raises(ValueError):
        calculate_doppler_shift(1e9, np.array([1, 2]), np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3]))
