from pathlib import Path
import yaml


def load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        project_root = Path(__file__).resolve().parents[2]
        path = project_root / "config.yaml"
    else:
        path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)