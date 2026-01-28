import numpy as np
from gnc_toolkit.actuators.actuator import Actuator

class Magnetorquer(Actuator):
    """
    Magnetorquer model.
    """
    def __init__(self, max_dipole=None, name="MTQ"):
        """
        Args:
            max_dipole (float): Maximum dipole moment [Am^2]. (Saturation)
            name (str): Name.
        """
        super().__init__(name=name, saturation=max_dipole)

    def command(self, dipole_cmd, **kwargs):
        """
        Args:
            dipole_cmd (float): Commanded dipole moment [Am^2].
            
        Returns:
            float: Delivered dipole moment [Am^2].
        """
        # Apply saturation
        return self.apply_saturation(dipole_cmd)
