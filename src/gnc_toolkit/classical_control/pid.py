import numpy as np
from typing import Optional, Union

class PID:
    """
    A generic PID controller with anti-windup logic.
    
    The controller output is calculated as:
    u(t) = Kp * e(t) + Ki * integral(e(t)) + Kd * derivative(e(t))
    """
    
    def __init__(
        self, 
        kp: float, 
        ki: float, 
        kd: float, 
        output_limits: Optional[tuple[float, float]] = None,
        anti_windup_method: str = "clamping"
    ):
        """
        Initialize the PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_limits: Optional tuple (min, max) for output saturation
            anti_windup_method: Method for anti-windup. Currently supports "clamping".
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.anti_windup_method = anti_windup_method
        
        self.integral_error = 0.0
        self.previous_error = 0.0
        self.reset()

    def reset(self):
        """Reset the internal state of the controller."""
        self.integral_error = 0.0
        self.previous_error = 0.0

    def update(self, error: float, dt: float) -> float:
        """
        Update the PID controller.
        
        Args:
            error: The current error signal (setpoint - measured)
            dt: Time step in seconds
            
        Returns:
            Control output
        """
        if dt <= 0:
            return 0.0
            
        # Proportional term
        p_term = self.kp * error
        
        # Integral term calculation (tentative)
        self.integral_error += error * dt
        i_term = self.ki * self.integral_error
        
        # Derivative term
        derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative
        
        # Calculate raw output
        output = p_term + i_term + d_term
        
        # Apply output limits and anti-windup
        if self.output_limits is not None:
            min_limit, max_limit = self.output_limits
            
            if output > max_limit:
                output_clamped = max_limit
            elif output < min_limit:
                output_clamped = min_limit
            else:
                output_clamped = output
            
            # Anti-windup (Clamping)
            # If output is saturated and error is driving it further into saturation,
            # stop integrating.
            if self.anti_windup_method == "clamping":
                if output != output_clamped:
                    # Classic clamping:
                    if (output > max_limit and error > 0) or (output < min_limit and error < 0):
                        # Revert integration
                        self.integral_error -= error * dt
            
            output = output_clamped
            
        self.previous_error = error
        return output
