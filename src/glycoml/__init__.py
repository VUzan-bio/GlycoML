"""GlycoML package."""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("glycoml")
except _metadata.PackageNotFoundError:
    __version__ = "0.0.0"

from . import phase1, phase2, shared

__all__ = ["phase1", "phase2", "shared", "__version__"]
