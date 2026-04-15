"""Phase 2: lectin-glycan interaction prediction.

Submodules are imported defensively so the package can be used even when
optional heavy dependencies (sklearn for ``baselines``, torch-geometric for
graph encoders, etc.) are not installed.
"""

from . import models, utils

for _name in ("baselines", "data", "eval", "scripts"):
    try:
        globals()[_name] = __import__(f"glycoml.phase2.{_name}", fromlist=["*"])
    except ImportError:  # pragma: no cover
        globals()[_name] = None

try:
    from .train import main as train  # noqa: F401
except ImportError:  # pragma: no cover
    train = None  # type: ignore[assignment]

__all__ = ["baselines", "data", "eval", "models", "scripts", "utils", "train"]
