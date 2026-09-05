"""
Configuration Management
Loads and merges YAML configuration files with dictionary access.
"""

from pathlib import Path
from typing import Any, Dict, Union
import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.
    
    Args:
        config_path: Path to the YAML file.
        
    Returns:
        Dictionary containing configuration parameters.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}
