# Models

Any [Anomalib model](https://anomalib.readthedocs.io/) can be used by name (`Padim`, `Patchcore`, `EfficientAd`,
`Dinomaly`, `AnomalyDino`, `InpFormer`, `Glass`, `WinClip`, ...), in Python and in the YAML config. SARIAD adds the
SAR-specific models below (generated from `SARIAD.models.MODELS_INFO`), the SAR pre-processing methods, and shares
one Gaussian base for feature-based models.

## SARIAD models

```{include} _generated/model_table.md
```

- **`SARATRX`**: the released SARATR-X checkpoint contains only the *encoder*; the decoder is trained on normal
  images (`freeze_encoder=True` trains just the decoder). It predicts multi-scale SAR gradient features of masked
  patches, and the anomaly map is the prediction error averaged over several random masks.
- **`YOLOAnomaly`**: any Ultralytics model works (`weights="yolov8n.pt" | "yolo11n.pt" | "yolo26n.pt" | "rtdetr-l.pt" |
  ...`). Layers are chosen by index into `YOLO(weights).model.model`; the defaults suit YOLOv8/YOLO11.
- **`PadimACE`**: needs anomalous *training* images (`signature_dir`). `whitening="reference"` reproduces the original
  PaDiM-ACE code; `"standard"` is textbook ACE. They differ, see the note in the API docs of
  `SARIAD.models.image.PadimACE.torch_model.PadimACEModel`.
- **`MSFA`**: only the *data-input* stage of the SARDet-100K method (filter augmentation) is reproduced. Its detector
  pretraining and model-migration stages train MMDetection networks and are not part of an anomaly detector.

## Pre-processing

`SARIAD.pre_processing`: `NLM` (non-local means), `MedianFilter`, `SARCNN` (learned despeckling, from the
[SAR-CNN](https://github.com/grip-unina/SAR-CNN) submodule; runs on the raw intensity) and `Default`.

## Candidate models from the literature review

Sources: the [satellite-image-deep-learning SAR list](https://github.com/satellite-image-deep-learning/techniques#sar)
(most entries are tools, not models). Repository facts (license, last push) were read from GitHub when this page was
written; the "Fit" column is a judgement from the repository descriptions, and none of these were run.

| Candidate | Task | License | Last push | Fit for anomaly detection |
|---|---|---|---|---|
| [SAR_CD_GKSNet](https://github.com/summitgao/SAR_CD_GKSNet) | Change detection (needs image pairs) | none | 2022 | Different task; would need bi-temporal data |
| [Pixel-wise segmentation, encoder-decoder + CRF](https://github.com/flyingshan/pixel-wise-segmentation-of-sar-imagery-using-encoder-decoder-network-and-fully-connected-crf) | Supervised segmentation | none | 2020 | Supervised baseline at best; notebook code |
| [XAI4SAR-PGIL](https://github.com/Alien9427/XAI4SAR-PGIL) | Patch-wise classification | none | 2023 | Backbone as feature extractor (like `YOLOAnomaly`) |
| [SARSeg (MP-ResNet)](https://github.com/DingLei14/SARSeg) | Supervised segmentation | none | 2021 | Backbone as feature extractor |
| [pytorch_self_supervised_learning](https://github.com/cattale93/pytorch_self_supervised_learning) | Self-supervised SAR/optical segmentation | MIT | 2026 | Most promising: permissive license, maintained |
| [Ship-Detection-...-SAR-Data](https://github.com/jasonmanesis/Ship-Detection-on-Remote-Sensing-Synthetic-Aperture-Radar-Data) | Ship detection (YOLOv5 on HRSID, notebooks) | MIT | 2022 | Covered by `YOLOAnomaly` (Ultralytics) |
| Clarifai SAR classification | Classification | closed source | n/a | Not integrable |
| [SARDet_100K / MSFA](https://github.com/zcablii/SARDet_100K) | SAR object detection | not stated | 2025 | Integrated (input stage), see above |
| [SARATR-X](https://github.com/waterdisappear/SARATR-X) | SAR foundation model | not stated | 2026 | Integrated as `SARATRX` |

Repositories without a license cannot be copied into SARIAD; integrate them as an optional dependency or git
submodule, or ask the authors.

## Adding a model

Implement a `torch_model.py` (an `nn.Module` returning an `InferenceBatch` in eval mode) and a
`lightning_model.py` (an `AnomalibModule`), register the class in `SARIAD/models/__init__.py` (`_MODELS`,
`MODELS_INFO`) so it is imported lazily, and add a forward-shape test that runs without downloads. For feature-based
models, `SARIAD.models.components.FeatureGaussianModel` needs only a feature extractor.
