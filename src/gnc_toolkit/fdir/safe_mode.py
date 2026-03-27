"""
Safe mode and fault detection logic for system mode transitions.
"""

import time
from collections.abc import Callable
from enum import Enum
from typing import Any, Optional


class SystemMode(Enum):
    NOMINAL = "NOMINAL"
    SAFE = "SAFE"
    RECOVERY = "RECOVERY"


class SafeModeCondition:
    """
    Trigger logic for Safe Mode state transitions.

    Monitors a specific fault condition and confirms it based on a 
    time-to-trigger threshold to avoid false positives.

    Parameters
    ----------
    check_fn : Callable[[], bool]
        Function returning True if the condition is currently violated.
    trigger_time_sec : float, optional
        Confirmation threshold (s). Default is 0.0 (immediate).
    """

    def __init__(self, check_fn: Callable[[], bool], trigger_time_sec: float = 0.0):
        """Initialize trigger condition."""
        self.check_fn = check_fn
        self.trigger_time_sec = trigger_time_sec
        self.violated_start_time: float | None = None

    def update(self) -> bool:
        """
        Evaluate condition and return True if confirmation time is exceeded.

        Returns
        -------
        bool
            True if safety trigger is active.
        """
        if self.check_fn():
            if self.violated_start_time is None:
                self.violated_start_time = time.time()

            elapsed = time.time() - self.violated_start_time
            return elapsed >= self.trigger_time_sec
        
        self.violated_start_time = None
        return False


class SafeModeLogic:
    """
    Finite State Machine for Spacecraft Mode Management (FDIR).

    Parameters
    ----------
    initial_mode : SystemMode, optional
        Starting mode. Default is NOMINAL.
    """

    def __init__(self, initial_mode: SystemMode = SystemMode.NOMINAL):
        """Initialize mode logic."""
        self.mode = initial_mode
        self.conditions: dict[str, SafeModeCondition] = {}
        self.history: list[dict[str, Any]] = []

    def add_condition(self, name: str, condition: SafeModeCondition) -> None:
        """
        Register a new safety trigger.

        Parameters
        ----------
        name : str
            Unique identifier for the condition.
        condition : SafeModeCondition
            Trigger implementation.
        """
        self.conditions[name] = condition

    def update(self) -> SystemMode:
        """
        Process all conditions and update system mode.

        Returns
        -------
        SystemMode
            Current operating mode.
        """
        if self.mode == SystemMode.SAFE:
            return self.mode

        for name, cond in self.conditions.items():
            if cond.update():
                self.mode = SystemMode.SAFE
                self.history.append(
                    {
                        "time": time.time(),
                        "from_mode": SystemMode.NOMINAL.value,
                        "to_mode": SystemMode.SAFE.value,
                        "reason": f"Condition {name} triggered",
                    }
                )
                break

        return self.mode

    def force_mode(self, mode: SystemMode, reason: str = "Commanded") -> None:
        """
        Force a mode transition (e.g., via ground command).

        Parameters
        ----------
        mode : SystemMode
            Target mode.
        reason : str, optional
            Rationale for the forced change.
        """
        old_mode = self.mode
        self.mode = mode
        self.history.append(
            {
                "time": time.time(),
                "from_mode": old_mode.value,
                "to_mode": mode.value,
                "reason": reason,
            }
        )
