import numpy as np
from gnc_toolkit.actuators.cmg import ControlMomentGyro

class VariableSpeedCMG(ControlMomentGyro):
    """
    Variable Speed Control Moment Gyro (VSCMG).
    Allows wheel speed to change (Reaction Wheel effect) while gimbaling (CMG effect).
    Torque = I_w * alpha * s + gimbal_rate * h * t
    """
    def __init__(self, wheel_inertia, gimbal_axis, spin_axis_init, name="VSCMG", 
                 max_gimbal_rate=None, max_wheel_torque=None):
        """
        Args:
            wheel_inertia (float): Moment of inertia of the wheel [kg*m^2].
            gimbal_axis (np.array): Gimbal axis (3,).
            spin_axis_init (np.array): Initial spin axis (3,).
            max_wheel_torque (float): Max torque for wheel acceleration.
        """
        # Initialize parent with dummy momentum (will update dynamically)
        super().__init__(wheel_momentum=0.0, gimbal_axis=gimbal_axis, 
                         spin_axis_init=spin_axis_init, name=name, 
                         max_gimbal_rate=max_gimbal_rate)
        
        self.inertia = wheel_inertia
        self.wheel_speed = 0.0
        self.max_wheel_torque = max_wheel_torque

    def command(self, cmd_vec, dt=None):
        """
        Calculate combined torque.
        
        Args:
            cmd_vec (tuple/list): (gimbal_rate_cmd, wheel_torque_cmd)
            dt (float, optional): Time step [s].
            
        Returns:
            np.array: Net torque vector [Nm] (3,).
        """
        g_rate_cmd, w_torque_cmd = cmd_vec
        
        # Apply saturation
        g_rate = self.apply_saturation(g_rate_cmd)
        if self.max_wheel_torque is not None:
            w_torque = np.clip(w_torque_cmd, -self.max_wheel_torque, self.max_wheel_torque)
        else:
            w_torque = w_torque_cmd
            
        # Update momentum magnitude for parent logic
        self.h_mag = self.inertia * self.wheel_speed
        
        # Current axes
        s, t = self.get_axes()
        
        # Net Torque = Torque_RW + Torque_CMG
        # T = w_torque * s + g_rate * (h * t)
        torque_vec = w_torque * s + g_rate * self.h_mag * t
        
        # Update states
        if dt is not None:
            self.gimbal_angle += g_rate * dt
            self.gimbal_angle = (self.gimbal_angle + np.pi) % (2 * np.pi) - np.pi
            self.wheel_speed += (w_torque / self.inertia) * dt
            
        return torque_vec

    def get_jacobian(self, angle=None):
        """
        Returns full Jacobian matrix mapping [g_rate, w_torque]^T to torque vector.
        
        Returns:
            np.array: (3, 2) matrix.
        """
        s, t = self.get_axes(angle)
        h = self.inertia * self.wheel_speed
        
        # T = [h*t , s] * [g_rate, w_torque]^T
        jac = np.zeros((3, 2))
        jac[:, 0] = h * t
        jac[:, 1] = s
        return jac
