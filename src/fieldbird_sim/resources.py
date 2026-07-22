from __future__ import annotations

from pathlib import Path
import shutil


PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"


def bundled_species_dir() -> Path:
    return PACKAGE_DATA_DIR / "species"


def bundled_calendar_path() -> Path:
    return PACKAGE_DATA_DIR / "configs" / "crop_calendar.csv"


def bundled_template_path() -> Path:
    return PACKAGE_DATA_DIR / "templates" / "run.yml"


def _relative_candidates(path: Path, *, base_dir: Path | None) -> list[Path]:
    if path.is_absolute():
        return [path]

    candidates: list[Path] = []
    if base_dir is not None:
        config_relative = Path(base_dir) / path
        candidates.append(config_relative)
    cwd_relative = Path.cwd() / path
    if cwd_relative not in candidates:
        candidates.append(cwd_relative)
    return candidates


def resolve_calendar_path(raw_path: str | Path, *, base_dir: Path | None = None) -> Path:
    path = Path(raw_path).expanduser()
    for candidate in _relative_candidates(path, base_dir=base_dir):
        if candidate.is_file():
            return candidate.resolve()

    if path.as_posix() in {"crop_calendar.csv", "configs/crop_calendar.csv"}:
        bundled = bundled_calendar_path()
        if bundled.is_file():
            return bundled

    raise FileNotFoundError(
        f"Crop calendar CSV not found: {raw_path}. "
        "Use an absolute path, a path relative to the config file, or the bundled default."
    )


def resolve_species_dir(
    species_name: str,
    *,
    configured_dir: str | Path | None = None,
    base_dir: Path | None = None,
) -> Path:
    parameter_name = f"species_{species_name}.csv"
    preference_names = (
        f"species_{species_name}_pref.csv",
        f"species_{species_name}_prefs.csv",
    )

    if configured_dir is not None:
        candidates = _relative_candidates(Path(configured_dir).expanduser(), base_dir=base_dir)
    else:
        candidates = [Path.cwd()]
        if base_dir is not None and Path(base_dir) not in candidates:
            candidates.append(Path(base_dir))
        candidates.append(bundled_species_dir())

    for candidate in candidates:
        if (candidate / parameter_name).is_file() and any(
            (candidate / name).is_file() for name in preference_names
        ):
            return candidate.resolve()

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Species data for {species_name!r} not found. Searched: {searched}. "
        "Run `fieldbird-sim init <directory>` to create editable parameter files."
    )


def create_starter_project(target_dir: Path, *, force: bool = False) -> list[Path]:
    target_dir = target_dir.expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "run.yml": bundled_template_path(),
        "crop_calendar.csv": bundled_calendar_path(),
    }
    for source in sorted(bundled_species_dir().glob("*.csv")):
        sources[source.name] = source

    existing = [target_dir / name for name in sources if (target_dir / name).exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing starter files: {names}. "
            "Use --force to replace all starter files."
        )

    written: list[Path] = []
    for name, source in sources.items():
        destination = target_dir / name
        shutil.copyfile(source, destination)
        written.append(destination)
    return written
