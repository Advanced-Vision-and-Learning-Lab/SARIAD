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
    "PadimACE": ".image.PadimACE",
    "MSFA": ".image.MFSA",  # the package directory keeps its historical spelling
}

__all__ = sorted(_MODELS)


def __getattr__(name: str):
    if name in _MODELS:
        return getattr(importlib.import_module(_MODELS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


# Documentation metadata of the SARIAD models (static, so it can be read without importing them).
# Anomalib's own models (Padim, Patchcore, EfficientAd, Dinomaly, ...) are available by name as well.
MODELS_INFO = {
    "SARATRX": {
        "summary": "HiViT masked autoencoder pretrained on SAR imagery (SARATR-X); scores the error of predicted SAR gradient features.",
        "paper": "https://arxiv.org/abs/2405.09365",
        "code": "https://github.com/waterdisappear/SARATR-X",
        "training": "decoder (and optionally encoder) trained on normal images; encoder pretrained",
        "requires": "SARATR-X git submodule and checkpoint (downloaded on first use)",
    },
    "YOLOAnomaly": {
        "summary": "Gaussian (PaDiM-style) anomaly model on intermediate features of an Ultralytics model (YOLOv8/11/26, RT-DETR, FastSAM, ...).",
        "paper": "https://docs.ultralytics.com/",
        "code": "https://github.com/ultralytics/ultralytics",
        "training": "none (statistics of normal images)",
        "requires": "ultralytics weights (downloaded on first use)",
    },
    "PadimACE": {
        "summary": "PaDiM with an adaptive cosine estimator scoring against target signatures from anomalous training images.",
        "paper": "https://arxiv.org/abs/2504.08049",
        "code": "https://github.com/Advanced-Vision-and-Learning-Lab/PaDiM-ACE",
        "training": "none (statistics of normal images + mean embedding of anomalous images)",
        "requires": "a folder of anomalous training images (`signature_dir`)",
    },
    "MSFA": {
        "summary": "PaDiM-style Gaussian model on backbone features of a filter-augmented input (raw + log-ratio edges, HOG, Canny).",
        "paper": "https://arxiv.org/abs/2403.06534",
        "code": "https://github.com/zcablii/SARDet_100K",
        "training": "none (statistics of normal images)",
        "requires": "-",
    },
}
