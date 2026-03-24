"""
Reaction Wheel (RW) actuator model with friction and momentum limits.
"""

import numpy as np

from gnc_toolkit.actuators.actuator import Actuator


class ReactionWheel(Actuator):
    """
    Reaction Wheel Actuator.
    Models torque generation with saturation and max speed limits.
    """

    def __init__(
        self,
        max_torque=None,
        max_momentum=None,
        inertia=None,
        name="RW",
        static_friction=0.0,
        viscous_friction=0.0,
        coulomb_friction=0.0,
    ):
        """
        Args:
            max_torque (float): Maximum torque [Nm]. (Saturation)
            max_momentum (float): Maximum angular momentum [Nms].
            inertia (float): Moment of inertia about spin axis [kg*m^2].
            name (str): Name.
            static_friction (float): Static friction torque [Nm].
            viscous_friction (float): Viscous friction coefficient [Nm/(rad/s)].
            coulomb_friction (float): Coulomb friction torque [Nm].
        """
        super().__init__(name=name, saturation=max_torque)
        self.max_momentum = max_momentum
        self.inertia = inertia
        self.static_friction = static_friction
        self.viscous_friction = viscous_friction
        self.coulomb_friction = coulomb_friction

    def command(self, torque_cmd, current_speed=0.0):
        """
        Calculate delivered torque.

        Args:
            torque_cmd (float): Commanded torque [Nm].
            current_speed (float): Current wheel speed [rad/s].

        Returns
        -------
            float: Delivered torque [Nm].
        """
        # Apply deadband - Default none.
        torque = self.apply_deadband(torque_cmd)

        # Apply motor torque limit (saturation)
        torque = self.apply_saturation(torque)

        # Apply friction effects
        friction_torque = 0.0
        if abs(current_speed) < 1e-4:  # "Zero" speed
            # Static friction resists the command
            if abs(torque) < self.static_friction:
                torque = 0.0
            else:
                torque -= np.sign(torque) * self.static_friction
        else:
            # Viscous and Coulomb friction
            friction_torque = (
                self.viscous_friction * current_speed
                + self.coulomb_friction * np.sign(current_speed)
            )
            torque -= friction_torque

        # Check momentum saturation (Speed limit)
        if self.inertia is not None and self.max_momentum is not None:
            current_momentum = self.inertia * current_speed

            if (current_momentum >= self.max_momentum and torque > 0) or (
                current_momentum <= -self.max_momentum and torque < 0
            ):
                torque = 0

        return torque
