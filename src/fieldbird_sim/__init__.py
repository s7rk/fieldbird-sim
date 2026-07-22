from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("fieldbird-sim")
except PackageNotFoundError:  # pragma: no cover - source tree without installation metadata
    __version__ = "0.1.0"


__all__ = ["__version__"]
