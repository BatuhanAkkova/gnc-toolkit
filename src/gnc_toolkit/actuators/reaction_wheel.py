import numpy as np
from gnc_toolkit.actuators.actuator import Actuator

class ReactionWheel(Actuator):
    """
    Reaction Wheel Actuator.
    Models torque generation with saturation and max speed limits.
    """
    def __init__(self, max_torque=None, max_momentum=None, inertia=None, name="RW"):
        """
        Args:
            max_torque (float): Maximum torque [Nm]. (Saturation)
            max_momentum (float): Maximum angular momentum [Nms].
            inertia (float): Moment of inertia about spin axis [kg*m^2].
            name (str): Name.
        """
        super().__init__(name=name, saturation=max_torque)
        self.max_momentum = max_momentum
        self.inertia = inertia

    def command(self, torque_cmd, current_speed=0.0):
        """
        Calculate delivered torque.
        
        Args:
            torque_cmd (float): Commanded torque [Nm].
            current_speed (float): Current wheel speed [rad/s].
            
        Returns:
            float: Delivered torque [Nm].
        """
        # Apply deadband - Default none.
        torque = self.apply_deadband(torque_cmd)
        
        # Apply torque saturation
        torque = self.apply_saturation(torque)
        
        # Check momentum saturation (Speed limit)
        if self.inertia is not None and self.max_momentum is not None:
            current_momentum = self.inertia * current_speed

            if current_momentum >= self.max_momentum and torque > 0:
                torque = 0
            elif current_momentum <= -self.max_momentum and torque < 0:
                torque = 0
        
        return torque
