from __future__ import annotations

from importlib import metadata
import platform

from fieldbird_sim import __version__


RUNTIME_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyyaml": "PyYAML",
    "shapely": "shapely",
    "geopandas": "geopandas",
    "pyproj": "pyproj",
    "pyogrio": "pyogrio",
    "matplotlib": "matplotlib",
}


def installed_dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for label, distribution in RUNTIME_DISTRIBUTIONS.items():
        try:
            versions[label] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[label] = None
    return versions


def runtime_metadata() -> dict[str, object]:
    return {
        "fieldbird_sim_version": __version__,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "dependencies": installed_dependency_versions(),
    }
