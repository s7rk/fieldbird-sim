from __future__ import annotations

import os
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .paths import ensure_dir, project_root


@dataclass
class RunResult:
    ok: bool
    error: Optional[BaseException] = None


def _patch_pandas_parent_dir_creation() -> None:
    """
    Patch pandas.io.common.check_parent_directory so it creates the folder
    instead of raising OSError. This prevents legacy code from crashing
    if a parent folder was not created earlier for any reason.
    """
    try:
        import pandas.io.common as pio  # type: ignore
    except Exception:
        return

    orig = getattr(pio, "check_parent_directory", None)
    if not callable(orig):
        return

    def _check_parent_directory(path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    pio.check_parent_directory = _check_parent_directory  # type: ignore


def run_legacy(
    entry: Path,
    cwd: Optional[Path] = None,
    patch_pandas_parent_dir: bool = True,
) -> RunResult:
    """
    Execute legacy_sim/trash.py via runpy without modifying the file.
    Keeps outputs identical to legacy behavior (relative paths etc.).
    """
    entry = Path(entry).resolve()
    if not entry.exists():
        raise FileNotFoundError(f"Legacy entry not found: {entry}")

    if cwd is None:
        cwd = project_root()

    cwd = Path(cwd).resolve()
    ensure_dir(cwd / "outputs")  # legacy writes to outputs/...

    old_cwd = Path.cwd()
    try:
        os.chdir(cwd)

        if patch_pandas_parent_dir:
            _patch_pandas_parent_dir_creation()

        ensure_dir(cwd / "outputs")
        runpy.run_path(str(entry), run_name="__main__")
        return RunResult(ok=True)

    except BaseException as e:
        return RunResult(ok=False, error=e)

    finally:
        os.chdir(old_cwd)
