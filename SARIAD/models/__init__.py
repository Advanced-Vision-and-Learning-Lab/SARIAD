"""SARIAD models.

Models are imported lazily so that ``import SARIAD.models`` works even when the optional
third-party submodules a model builds on (SARATR-X, SARDet_100K, ...) have not been checked
out (``git submodule update --init``). The submodule is only needed once the model is used.
"""

import importlib

# model class name -> module (relative to this package) that defines it
_MODELS = {
    "SARATRX": ".image.SARATRX",
    "YOLOAnomaly": ".image.YOLO",
}

__all__ = sorted(_MODELS)


def __getattr__(name: str):
    if name in _MODELS:
        return getattr(importlib.import_module(_MODELS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
