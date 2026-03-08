import numpy as np
from gnc_toolkit.actuators.actuator import Actuator

class ControlMomentGyro(Actuator):
    """
    Control Moment Gyro (CMG) Actuator.
    Models torque produced by changing the orientation of a constant-speed momentum wheel.
     Torque = gimbal_rate x momentum
    """
    def __init__(self, wheel_momentum, gimbal_axis, spin_axis_init, name="CMG", max_gimbal_rate=None):
        """
        Args:
            wheel_momentum (float): Angular momentum magnitude [Nms].
            gimbal_axis (np.array): Unit vector of the gimbal axis (3,).
            spin_axis_init (np.array): Unit vector of the initial spin axis (3,).
            name (str): Name.
            max_gimbal_rate (float): Maximum gimbal angular velocity [rad/s].
        """
        super().__init__(name=name, saturation=max_gimbal_rate)
        self.h_mag = wheel_momentum
        self.g_axis = np.array(gimbal_axis) / np.linalg.norm(gimbal_axis)
        self.s_axis = np.array(spin_axis_init) / np.linalg.norm(spin_axis_init)
        
        # Verify orthogonality (approximately)
        if abs(np.dot(self.g_axis, self.s_axis)) > 1e-6:
            # Re-orthogonalize s_axis
            self.s_axis = self.s_axis - np.dot(self.s_axis, self.g_axis) * self.g_axis
            self.s_axis /= np.linalg.norm(self.s_axis)

        self.t_axis = np.cross(self.g_axis, self.s_axis)  # Transverse axis
        self.gimbal_angle = 0.0

    def get_axes(self, angle=None):
        """
        Get the current spin and transverse axes for a given gimbal angle.
        
        Args:
            angle (float, optional): Gimbal angle [rad]. If None, uses current.
            
        Returns:
            tuple: (spin_axis, transverse_axis)
        """
        theta = angle if angle is not None else self.gimbal_angle
        # Rotation about gimbal axis
        # Rodrigues' rotation formula or just sin/cos since it's 2D in the plane
        s = self.s_axis * np.cos(theta) + self.t_axis * np.sin(theta)
        t = np.cross(self.g_axis, s)
        return s, t

    def command(self, gimbal_rate_cmd, dt=None):
        """
        Calculate torque produced by gimbal rate.
        
        Args:
            gimbal_rate_cmd (float): Commanded gimbal rate [rad/s].
            dt (float, optional): Time step [s] to update gimbal angle.
            
        Returns:
            np.array: Torque vector [Nm] (3,).
        """
        # Apply rate limits
        g_rate = self.apply_saturation(gimbal_rate_cmd)
        
        # Current axes
        s, t = self.get_axes()
        
        # Torque = omega_g x h = (g_rate * g_axis) x (h_mag * s)
        # Torque = g_rate * h_mag * (g x s) = g_rate * h_mag * t
        torque_vec = g_rate * self.h_mag * t
        
        # Update gimbal angle
        if dt is not None:
            self.gimbal_angle += g_rate * dt
            # Normalize angle to [-pi, pi]
            self.gimbal_angle = (self.gimbal_angle + np.pi) % (2 * np.pi) - np.pi
            
        return torque_vec

    def get_torque_jacobian(self, angle=None):
        """
        Returns the Jacobian mapping gimbal rate to torque: T = A * g_rate.
        
        Args:
            angle (float, optional): Angle to evaluate at.
            
        Returns:
            np.array: (3,1) mapping matrix.
        """
        _, t = self.get_axes(angle)
        return (self.h_mag * t).reshape(3, 1)
