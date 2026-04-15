"""GlycoML package."""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("glycoml")
except _metadata.PackageNotFoundError:
    __version__ = "0.0.0"

# Subpackages are imported defensively so that downstream code can use any
# phase without needing to install the dependencies of the others.
from . import shared  # noqa: F401

for _name in ("phase1", "phase2", "phase3"):
    try:
        __import__(f"glycoml.{_name}")
    except ImportError:  # pragma: no cover
        pass

__all__ = ["phase1", "phase2", "phase3", "shared", "__version__"]
