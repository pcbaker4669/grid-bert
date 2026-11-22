# ============================================================
# GridBERT v0.3 - Configuration Loader
#
# Purpose:
#   Provide shared configuration and path-handling functions
#   used by GridBERT scripts.
#
# Pipeline position:
#   Shared project utility.
#   This file is not a separate processing stage. It is imported
#   by scripts throughout the GridBERT workflow.
#
# Functions:
#   load_config()
#       Loads config.yaml from the project root.
#
#   data_path(config, key)
#       Resolves a configured path relative to data_root unless
#       the configured value is already an absolute path.
#
# Main input:
#   config.yaml
#
# Main output:
#   Loaded configuration values and resolved data paths for
#   other GridBERT scripts.
# ============================================================

from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config.yaml"


def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def data_path(config, key):
    root = Path(config["paths"]["data_root"])
    value = Path(config["paths"][key])

    if value.is_absolute():
        return value

    return root / value