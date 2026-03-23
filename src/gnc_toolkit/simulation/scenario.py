import json
import os
from pathlib import Path
from typing import Dict, Any, Union

class ScenarioConfig:
    """
    Scenario configuration module.
    Responsible for parsing reproducible mission setups from JSON (or YAML if available).
    """

    def __init__(self, filename: Union[str, Path]):
        """
        Initialize the scenario configuration.

        Parameters
        ----------
        filename : str or Path
            Path to the JSON or YAML configuration file.
        """
        self.filename = Path(filename)
        self.config: Dict[str, Any] = {}
        self.load()

    def load(self):
        """Loads the configuration file."""
        if not self.filename.exists():
            raise FileNotFoundError(f"Scenario configuration file not found: {self.filename}")

        ext = self.filename.suffix.lower()
        if ext == ".json":
            with open(self.filename, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        elif ext in [".yaml", ".yml"]:
            try:
                import yaml
                with open(self.filename, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required to parse YAML scenarios. Install it with `pip install pyyaml`.")
        else:
            raise ValueError(f"Unsupported configuration file extension: {ext}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value by key. Supports dot notation for nested keys.

        Parameters
        ----------
        key : str
            Configuration key (e.g., 'satellite.mass').
        default : Any
            Default value if key is not found.

        Returns
        -------
        Any
            The configuration value.
        """
        keys = key.split('.')
        current = self.config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current
