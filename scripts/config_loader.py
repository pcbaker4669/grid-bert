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