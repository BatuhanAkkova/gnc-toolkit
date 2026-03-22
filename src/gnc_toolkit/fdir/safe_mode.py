"""
Safe mode and fault detection logic for system mode transitions.
"""

import time
from enum import Enum
from typing import Dict, Any, Callable, List

class SystemMode(Enum):
    NOMINAL = "NOMINAL"
    SAFE = "SAFE"
    RECOVERY = "RECOVERY"

class SafeModeCondition:
    """
    Defines a condition that triggers safe mode.
    """
    def __init__(self, check_fn: Callable[[], bool], trigger_time_sec: float = 0.0):
        """
        Initialize a safe mode condition.
        
        Args:
            check_fn: A function that returns True if the condition is violated (faulty state)
            trigger_time_sec: Duration the condition must be violated before triggering
        """
        self.check_fn = check_fn
        self.trigger_time_sec = trigger_time_sec
        self.violated_start_time = None
        
    def update(self) -> bool:
        """
        Update the condition state and check if it should trigger.
        
        Returns:
            True if safety condition triggered, False otherwise
        """
        if self.check_fn():
            if self.violated_start_time is None:
                self.violated_start_time = time.time()
            
            elapsed = time.time() - self.violated_start_time
            if elapsed >= self.trigger_time_sec:
                return True
        else:
            self.violated_start_time = None
            
        return False

class SafeModeLogic:
    """
    Manages system mode transitions based on fault conditions.
    """
    def __init__(self, initial_mode: SystemMode = SystemMode.NOMINAL):
        """
        Initialize safe mode logic.
        """
        self.mode = initial_mode
        self.conditions: Dict[str, SafeModeCondition] = {}
        self.history: List[Dict[str, Any]] = []
        
    def add_condition(self, name: str, condition: SafeModeCondition):
        """
        Add a trigger condition.
        """
        self.conditions[name] = condition
        
    def update(self) -> SystemMode:
        """
        Update mode logic based on conditions.
        
        Returns:
            Current SystemMode
        """
        if self.mode == SystemMode.SAFE:
            return self.mode
            
        for name, cond in self.conditions.items():
            if cond.update():
                self.mode = SystemMode.SAFE
                self.history.append({
                    "time": time.time(),
                    "from_mode": SystemMode.NOMINAL.value,
                    "to_mode": SystemMode.SAFE.value,
                    "reason": f"Condition {name} triggered"
                })
                break
                
        return self.mode

    def force_mode(self, mode: SystemMode, reason: str = "Commanded"):
        """
        Force mode transition.
        """
        old_mode = self.mode
        self.mode = mode
        self.history.append({
            "time": time.time(),
            "from_mode": old_mode.value,
            "to_mode": mode.value,
            "reason": reason
        })
