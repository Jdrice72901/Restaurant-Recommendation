from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml and resolve its paths against the repo root."""
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for section, key in [("data", "raw_dir"), ("data", "processed_dir"), ("index", "dir")]:
        p = Path(cfg[section][key])
        cfg[section][key] = p if p.is_absolute() else REPO_ROOT / p
    return cfg
