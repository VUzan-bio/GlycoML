"""Phase 1: antibody N-glycosylation site prediction.

Submodules are imported defensively so that the package can be used even when
optional heavy dependencies (torch-geometric for ``fc_engineering``, etc.) are
not installed.
"""

from . import models, pipeline, utils

try:  # optional: requires torch-geometric
    from . import fc_engineering  # noqa: F401
except ImportError:  # pragma: no cover
    fc_engineering = None  # type: ignore[assignment]

try:
    from . import scripts  # noqa: F401
except ImportError:  # pragma: no cover
    scripts = None  # type: ignore[assignment]

try:
    from .train import main as train  # noqa: F401
except ImportError:  # pragma: no cover
    train = None  # type: ignore[assignment]

__all__ = ["fc_engineering", "models", "pipeline", "scripts", "utils", "train"]
