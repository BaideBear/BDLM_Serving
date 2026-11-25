"""Only compile controller definitions from the exact reviewed upstream files."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@lru_cache
def control_directory(name):
    lock = json.loads((ROOT / "third_party/controls.json").read_text())[name]
    path = ROOT / "third_party" / name / lock["directory"]
    for filename, expected in lock["files"].items():
        file = path / filename
        if not file.is_file():
            raise FileNotFoundError(f"{name} is missing; run git submodule update --init")
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{name}/{filename} differs from the archived controller")
    return path
