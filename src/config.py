"""
Configuration loader for the ODD exit RL project.

Loads YAML configuration files and provides a simple dictionary interface.
"""

import yaml


def load_config(path: str) -> dict:
    """
    Load a YAML configuration file.
    
    Args:
        path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
