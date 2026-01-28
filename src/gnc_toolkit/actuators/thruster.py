import numpy as np
from gnc_toolkit.actuators.actuator import Actuator

class Thruster(Actuator):
    """
    Base Thruster model.
    Produces thrust Force.
    """
    def __init__(self, max_thrust=1.0, min_impulse_bit=0.0, isp=None, name="Thruster"):
        """
        Args:
            max_thrust (float): Maximum thrust [N].
            min_impulse_bit (float): Minimum impulse bit [Ns].
                                     Actuator will not fire if requested impulse < MIB (unless 0).
            isp (float): Specific Impulse [s].
        """
        super().__init__(name=name, saturation=max_thrust, deadband=None)
        self.max_thrust = max_thrust
        self.min_impulse_bit = min_impulse_bit
        self.isp = isp

    def command(self, thrust_cmd, dt=None, **kwargs):
        """
        Args:
            thrust_cmd (float): Commanded thrust [N].
            dt (float, optional): Time step duration [s]. Required for MIB checks.
            
        Returns:
            float: Delivered thrust [N].
        """
        # Saturation (clip to max thrust)
        thrust = self.apply_saturation(thrust_cmd)
        
        # Minimum Impulse Bit Logic
        # If dt is provided, check if the requested impulse is possible.
        if dt is not None and self.min_impulse_bit > 0 and abs(thrust) > 1e-9:
            requested_impulse = abs(thrust) * dt
            if requested_impulse < self.min_impulse_bit:
                # Deadband behavior for impulses < MIB
                thrust = 0.0
                
        return thrust

    def get_mass_flow(self, thrust):
        """Calculate mass flow rate for a given thrust."""
        if self.isp and self.isp > 0:
            g0 = 9.80665
            return thrust / (self.isp * g0)
        return 0.0


class ChemicalThruster(Thruster):
    """
    Chemical Thruster.
    Models On/Off behavior or PWM-averaged thrust.
    """
    def __init__(self, max_thrust=10.0, isp=300.0, min_on_time=0.010, name="ChemThruster"):
        """
        Args:
            min_on_time (float): Minimum valve open time [s].
            (Calculates MIB = max_thrust * min_on_time)
        """
        self.min_on_time = min_on_time
        mib = max_thrust * min_on_time
        super().__init__(max_thrust=max_thrust, isp=isp, min_impulse_bit=mib, name=name)
    
    def command(self, thrust_cmd, dt=None, **kwargs):
        """
        Considers PWM constraints.
        If the commanded thrust implies an on-time < min_on_time, it is zeroed.
        """
        # Allow proportional command (representing average thrust via PWM).
        thrust = super().command(thrust_cmd, dt=dt, **kwargs)
        
        if dt is not None and self.min_on_time > 0 and abs(thrust) > 1e-9:
            # required_on_time = (thrust / self.max_thrust) * dt
            required_on_time = (abs(thrust) / self.max_thrust) * dt
            
            if required_on_time < self.min_on_time:
                thrust = 0.0
                
        return thrust


class ElectricThruster(Thruster):
    """
    Electric Thruster (e.g. Hall Effect, Ion).
    Power-limited.
    """
    def __init__(self, max_thrust=0.1, isp=1500.0, power_efficiency=0.6, name="ElecThruster"):
        """
        Args:
            power_efficiency (float): Electrical to Jet power efficiency (eta).
        """
        super().__init__(max_thrust=max_thrust, isp=isp, name=name)
        self.power_efficiency = power_efficiency
        self.g0 = 9.80665

    def get_power_consumption(self, thrust):
        """Calculate required electrical power [W]."""
        # P_in = Thrust * ve / (2 * eta)
        # ve = Isp * g0
        if self.power_efficiency <= 0: return float('inf')
        
        ve = self.isp * self.g0
        power = (thrust * ve) / (2 * self.power_efficiency)
        return power
