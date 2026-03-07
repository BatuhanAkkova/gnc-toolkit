import numpy as np
from gnc_toolkit.sensors.sensor import Sensor
from gnc_toolkit.sensors.gyroscope import Gyroscope

class Accelerometer(Sensor):
    """
    Accelerometer sensor model.
    Measures non-gravitational acceleration: a_meas = a_true + bias + noise
    """
    def __init__(self, noise_std=0.0, bias=None, scale_factor=1.0, name="Accelerometer"):
        """
        Args:
            noise_std (float): Standard deviation of measurement noise [m/s^2].
            bias (np.ndarray): Constant bias vector [m/s^2].
            scale_factor (float): Scale factor error (1.0 = no error).
        """
        super().__init__(name)
        self.noise_std = noise_std
        self.bias = bias if bias is not None else np.zeros(3)
        self.scale_factor = scale_factor

    def measure(self, true_accel, **kwargs):
        """
        Args:
            true_accel (np.ndarray): True non-gravitational acceleration [m/s^2].
        """
        noise = np.random.normal(0, self.noise_std, 3)
        measured_accel = self.scale_factor * true_accel + self.bias + noise
        return measured_accel

class IMU(Sensor):
    """
    Inertial Measurement Unit (IMU) combining Gyroscope and Accelerometer.
    """
    def __init__(self, gyro_params=None, accel_params=None, name="IMU"):
        """
        Args:
            gyro_params (dict): Parameters for Gyroscope initialization.
            accel_params (dict): Parameters for Accelerometer initialization.
        """
        super().__init__(name)
        gyro_p = gyro_params if gyro_params is not None else {}
        accel_p = accel_params if accel_params is not None else {}
        
        self.gyro = Gyroscope(**gyro_p)
        self.accel = Accelerometer(**accel_p)

    def measure(self, true_omega, true_accel, **kwargs):
        """
        Args:
            true_omega (np.ndarray): True angular velocity [rad/s].
            true_accel (np.ndarray): True non-gravitational acceleration [m/s^2].
            
        Returns:
            tuple: (measured_omega, measured_accel)
        """
        meas_omega = self.gyro.measure(true_omega, **kwargs)
        meas_accel = self.accel.measure(true_accel, **kwargs)
        return meas_omega, meas_accel
