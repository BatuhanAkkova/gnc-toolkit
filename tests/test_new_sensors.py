import pytest
import numpy as np
from gnc_toolkit.sensors.gnss_receiver import GNSSReceiver
from gnc_toolkit.sensors.imu import Accelerometer, IMU
from gnc_toolkit.sensors.sun_sensor_array import CoarseSunSensorArray
from gnc_toolkit.sensors.horizon_sensor import HorizonSensor
from gnc_toolkit.sensors.altimeter import Altimeter
from gnc_toolkit.sensors.lidar import Lidar
from gnc_toolkit.sensors.camera import Camera
from gnc_toolkit.sensors.star_catalog import StarCatalog

class TestNewSensors:
    def test_gnss_receiver(self):
        gnss = GNSSReceiver(pos_noise_std=0.0, vel_noise_std=0.0, 
                             pos_bias=np.array([1, 2, 3]))
        r_true = np.array([7000e3, 0, 0])
        v_true = np.array([0, 7.5e3, 0])
        r_meas, v_meas = gnss.measure(r_true, v_true)
        assert np.allclose(r_meas, r_true + np.array([1, 2, 3]))
        assert np.allclose(v_meas, v_true)

    def test_accelerometer(self):
        acc = Accelerometer(noise_std=0.0, bias=np.array([0.1, 0, 0]), scale_factor=1.1)
        a_true = np.array([1.0, 0.0, 0.0])
        a_meas = acc.measure(a_true)
        # 1.1 * 1.0 + 0.1 = 1.2
        assert np.isclose(a_meas[0], 1.2)

    def test_imu(self):
        imu = IMU(gyro_params={'noise_std': 0.0}, accel_params={'noise_std': 0.0})
        w_true = np.array([0.1, 0, 0])
        a_true = np.array([0, 0, 9.8])
        w_meas, a_meas = imu.measure(w_true, a_true)
        assert np.allclose(w_meas, w_true)
        assert np.allclose(a_meas, a_true)

    def test_css_array(self):
        css = CoarseSunSensorArray(noise_std=0.0)
        sun_vec = np.array([1.0, 0.0, 0.0])
        meas = css.measure(sun_vec)
        # Boresights: [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
        # Expected: 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
        assert np.isclose(meas[0], 1.0)
        assert np.all(meas[1:] == 0.0)

    def test_horizon_sensor(self):
        hs = HorizonSensor(noise_std=0.0)
        nadir = np.array([0, 0, 1.0])
        meas = hs.measure(nadir)
        assert np.allclose(meas, nadir)

    def test_altimeter(self):
        alt = Altimeter(noise_std=0.0, bias=10.0)
        meas = alt.measure(500e3)
        assert np.isclose(meas, 500010.0)

    def test_lidar(self):
        lidar = Lidar(range_noise_std=0.0, los_noise_std=0.0)
        rel_pos = np.array([10.0, 0.0, 0.0])
        r_meas, los_meas = lidar.measure(rel_pos)
        assert np.isclose(r_meas, 10.0)
        assert np.allclose(los_meas, np.array([1.0, 0.0, 0.0]))

    def test_camera(self):
        cam = Camera(focal_length=1.0, resolution=(1000, 1000), sensor_size=(1.0, 1.0))
        # Point at [0, 0, 10] -> center of image [500, 500]
        p = np.array([0, 0, 10.0])
        uv = cam.measure(p)
        assert np.allclose(uv, np.array([500, 500]))
        
        # Point at [0.1, 0, 10] -> u = 1.0 * 0.1 / 10 = 0.01. 
        # pixels_per_unit = 1000 / 1.0 = 1000. 
        # u = 0.01 * 1000 + 500 = 510.
        p2 = np.array([0.1, 0, 10.0])
        uv2 = cam.measure(p2)
        assert np.allclose(uv2, np.array([510, 500]))

    def test_star_catalog(self):
        cat = StarCatalog()
        cat.load_catalog(None) # Use dummy stars
        # Stars at [1,0,0], [0,1,0], [0,0,1], [-1,0,0]
        # Boresight [1,0,0], FOV 10 deg -> Only star 1
        stars = cat.get_stars_in_fov(np.array([1, 0, 0]), 10.0)
        assert len(stars) == 1
        assert stars[0]['id'] == 1
