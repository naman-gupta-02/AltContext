import json
from pathlib import Path

from .config import ROOT
from .text import build_context, concadia_fields

DATA_DIR = ROOT / "data"


def load_subset(name_or_path: str) -> list[dict]:
    p = Path(name_or_path)
    if not p.exists():
        p = DATA_DIR / "processed" / "subsets" / f"{name_or_path}.jsonl"
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def image_path(record: dict) -> Path:
    return DATA_DIR / "raw" / record["image_path"]


def record_context(record: dict, cfg: dict) -> str:
    """Bounded context for a manifest record. Never includes target_description."""
    return build_context(concadia_fields(record), cfg["context"])
