# Datasets

SARIAD turns SAR *target* datasets into anomaly detection benchmarks: the objects of interest (vehicles,
ships, ...) are the anomalies, and **normal** images are the same scenes without them.

## Available datasets

Generated from the code (`DATASET_INFO` in each datamodule).

```{include} _generated/dataset_table.md
```

All datamodules are [Anomalib `Folder`](https://anomalib.readthedocs.io/) datamodules and accept
`batch_size`, `num_workers`, `path` (dataset directory; default `<SARIAD_DATASETS_PATH or ./datasets>/<name>`)
and any other `Folder` argument, so the same parameters can be given from Python or from the YAML config.

## How normal data is generated

Most SAR datasets only contain images *with* targets. `SARIAD.utils.normal_gen` builds normal images in two
swappable steps ([API](source/SARIAD.utils)):

1. **Segment** the target and its shadow (`segment_target`): `classical` (KMeans + morphology) by default, or the
   experimental `birefnet` backend. Datasets that ship masks skip this.
2. **Fill** the mask with clutter from the rest of the image: `fill_from_background` (random background pixels) or
   `patch_inpaint` (random background patches, keeps more of the speckle structure).

`remove_target` refuses to produce a normal image when the segmentation covers most of the chip, because the
classical segmenter assumes dark clutter: on dB-scaled SAMPLE chips it selects the whole image, and the "normal" image
would still contain the target. Those chips are skipped and counted.

```{admonition} Known limitation
:class: warning
Filling with random pixels or patches decorrelates the speckle, leaving a faint disc where the target was, and a
model can learn to detect that. Compare against real target-free scenes where you have them.
```

## Train/test conventions

- **SAMPLE_PUBLIC** and **SARDet_100K** (new): the test anomalies come from a *separate* `test` split
  (SAMPLE: disjoint azimuth ranges), and training uses only normal images of the `train` split.
- **MSTAR** and **SSDD** (existing): the anomalous test images are the `{split}/anom` images of the configured
  split, i.e. the same chips whose target-free versions in `{split}/norm` are used for training. Keep this in mind when
  interpreting results, and treat it as a design decision to revisit.

## Candidate datasets (not yet integrated)

Links were checked when this page was written. "Fit" is a first impression from the dataset pages, not an evaluation.

| Dataset | What it is | Link | License | Fit for SARIAD |
|---|---|---|---|---|
| ATRNet-STAR | Large fine-grained SAR vehicle dataset and benchmark | [GitHub](https://github.com/waterdisappear/ATRNet-STAR) | Apache-2.0 (code) | Same recipe as MSTAR/SAMPLE (vehicle = anomaly); check chip size and scenes |
| TenGeoP-SARwv | Labeled Sentinel-1 wave-mode imagery of ten geophysical phenomena | [SEANOE](https://www.seanoe.org/data/00456/56796/) | see page | Natural-scene anomalies (e.g. one phenomenon as normal); no target masks |
| xView3-SAR | Sentinel-1 SAR for detecting illegal fishing vessels (large scenes) | [xView3](https://iuu.xview.us/) | see page | Vessel = anomaly on large tiles; needs tiling |
| SpaceNet 6 | Multi-sensor all-weather mapping (SAR + optical building footprints) | [SpaceNet](https://spacenet.ai/sn6-challenge/) | see page | Footprint-level masks exist; anomaly definition is unclear |
| QXS-SAROPT | Paired SAR/optical patches | [GitHub](https://github.com/yaoxu008/QXS-SAROPT) | none stated | Cross-modal pairs; no anomaly labels |

Anomalib itself ships more (non-SAR) datamodules, e.g. `BMAD` (the benchmark this project was inspired by),
`MVTecAD2`, `RealIAD`, `VAD`, `AutoVI` and `Kaputt`.

## Adding a dataset

1. Subclass `anomalib.data.Folder` in `SARIAD/datasets/datamodules/image/<name>/<name>.py`, following
   `sample_public.py`: download with `fetch_blob`, generate `{train,test}/{anom,norm,masks}` with `normal_gen` if
   needed, accept `path`, `batch_size`, `num_workers` and `**folder_kwargs`.
2. Add a `DATASET_INFO` dict (it feeds the table above) and register the class in `SARIAD/datasets/__init__.py`.
3. Add a generation test under `tests/` that runs on a tiny synthetic tree.
