import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "configs/default.json") -> dict:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = ROOT / p
    with open(p) as f:
        return json.load(f)
