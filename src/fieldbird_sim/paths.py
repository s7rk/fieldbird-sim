from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    # assuming this file lives in src/fieldbird_sim/paths.py
    return Path(__file__).resolve().parents[2]


def legacy_entry(default: str = "legacy_sim/trash.py") -> Path:
    return (project_root() / default).resolve()


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
