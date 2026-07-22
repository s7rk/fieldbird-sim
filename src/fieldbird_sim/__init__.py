from importlib import metadata


def _read_version() -> str:
    try:
        return metadata.version("fieldbird-sim")
    except metadata.PackageNotFoundError:
        return "0+unknown"


__version__ = _read_version()


__all__ = ["__version__"]
