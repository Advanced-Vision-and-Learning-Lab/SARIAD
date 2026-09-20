# Handoff: closing the open SARIAD issues

Written for the next Claude Code session (or person) picking this up on a machine **with a GPU**.
Everything below was done on a CPU-only WSL box with no datasets, so the GPU/real-data validation is the
main remaining work. Read "What was and was not verified" before trusting any number.

- **Branch:** `fix/open-issues` (branched from `main` at `1f9eae6`), 15 commits. One commit per issue with
  `Fixes #N` in the message (they auto-close when merged into the default branch). **It was NOT pushed**: the WSL
  box had no GitHub credentials. The repo lives on the Windows disk (`C:\Users\jpeep\Documents\SARIAD`), so push from
  Windows: `git push -u origin fix/open-issues`. **No PR has been opened and nothing was commented on or closed on
  GitHub** (`gh` is not installed there).
- **Issues:** https://github.com/Advanced-Vision-and-Learning-Lab/SARIAD/issues (17 were open).
- **Reference paper:** https://arxiv.org/abs/2504.08115 (SARIAD). PaDiM-ACE: https://arxiv.org/abs/2504.08049.

## Getting set up

```bash
git fetch && git checkout fix/open-issues
git submodule update --init          # SARATR-X, SAR-CNN, SARDet_100K (URLs were changed from SSH to HTTPS)
uv venv --python 3.13 && source .venv/bin/activate
uv pip install torch torchvision     # CUDA build for the GPU box, e.g. --index-url https://download.pytorch.org/whl/cu126
uv pip install -e ".[dev,docs]"      # anomalib>=2.6.2, ultralytics>=8.4
MLFLOW_DISABLE_AGENT_HINT=1 pytest   # ~70 offline tests (74 collected at the last full run, 2 added since); 5 gated ones skip
```

The pretrained SARATR-X checkpoint (`mae_hivit_base_1600ep.pth`, Google Drive id in
`SARIAD/models/image/SARATRX/torch_model.py`) downloads into anomalib's pretrained-weights cache on first use.
Set `SARIAD_TEST_SARATRX_CHECKPOINT=/path/to/it` and `SARIAD_TEST_NETWORK=1` to run the gated tests.

## What was done, per issue

| Issue | Status | Commit subject | Notes |
|---|---|---|---|
| #16 conda env | done | "Update environment to anomalib 2.6.2 ..." | env file mirrors pyproject; deps trimmed; archive extraction hardened against path traversal |
| #15 config params | done | "Config: consistent dataset params ..." | runner rewritten, `--dry-run` validates a YAML against real constructor signatures |
| #6 SAR-CNN | done | "SAR-CNN: lazy weight loading ..." | was breaking *every* preprocessor import on CPU-only machines |
| #9 SARATR-X | done, **needs GPU/real-data check** | "SARATRX: fix anomaly map ..." | see "SARATRX findings" |
| #32 YOLO | done | "YOLO: implement a Gaussian anomaly model ..." | hooks on Ultralytics layers + shared PaDiM-style Gaussian |
| #28 / #39 MSFA | partial by design | "MSFA: filter-augmented anomaly detector ..." | only the data-input stage of the SARDet-100K method |
| #2 / #4 / #8 datasets, normal data | done, real downloads not run | "Datasets: shared normal-data generation ..." | `SARIAD/utils/normal_gen.py` |
| #17 inferencer | done | "Add a streaming torchmetrics Inferencer ..." | pixel AUROC was computed from the binarized mask |
| #14 cleanup | done | "Cleanup: ...", README rewrite | dead `main.py` and 3 legacy models deleted |
| #7 dataset docs | **written, never built** | docs commit | see "Open items" #1 |
| #5 / #27 / #43 research | done as docs pages | docs commit | `docs/datasets.md`, `docs/models.md` (facts checked via GitHub API, nothing was run) |
| #19 overall figure | exists, **stale** | untouched | `figs/overall.svg` lists only "SARATR-X (WIP)" etc.; text is outlined paths so it can't be edited in place |
| (extra) PaDiM-ACE | done | "Add PaDiM-ACE ..." | requested mid-session; see "PaDiM-ACE" |

Also: `.gitattributes` (LF), publish workflow now only publishes on tags and runs `pytest` first, `sariad`
console script, test suite (`tests/`).

## What was and was not verified

**Run on CPU with synthetic data (real code paths, offline):**
the whole runner -> Engine -> metrics pipeline for Padim, PadimACE, YOLOAnomaly, MSFA, SARATRX (frozen encoder);
metrics vs the old implementation and vs scikit-learn; normal-data generation vs the original MSTAR/SSDD code
(identical masks); SAMPLE generation on 11 *real* chips downloaded individually; SARATRX with the **real checkpoint**
(its anomaly error equals the MAE's own training loss for a fixed mask, exactly).

**Not run / only reasoned about:**
- **Any result on real SAR data.** Synthetic AUROCs in the commit messages/discussion only show wiring. No claim is
  made about SARATRX/YOLO/MSFA/ACE quality on MSTAR/HRSID/SSDD.
- The full SAMPLE download (1.5 GB) and the SARDet-100K Kaggle download (tens of GB, needs credentials); SARDet
  generation was tested on a synthetic COCO fixture only, and its patch inpainting may be slow on 800x800 images.
- The BiRefNet segmentation backend (`normal_gen.segment_birefnet`): written, never executed (1 GB download,
  `trust_remote_code=True`). Treat as experimental.
- **The last edits were not re-tested** (the environment disappeared): the `MODELS_INFO` / `DATASETS_INFO`
  metadata, `docs/`, `.readthedocs.yaml`, README. Run `pytest` first.
- The `uv.lock` was regenerated before the `docs` extra was trimmed (`nbsphinx` removed): run `uv lock`.

## Open items, in priority order

1. **Build the docs** (#7). `docs/source/*.rst` are stale except `SARIAD.rst`. Regenerate and build:
   ```bash
   sphinx-apidoc -f -o docs/source SARIAD SARIAD/models/image/SARATRX/SARATRX \
     SARIAD/pre_processing/SARCNN/SARCNN_SRC SARIAD/models/image/MFSA/SARDet_100K SARIAD/results
   sphinx-build -b html docs docs/_build/html      # try -W --keep-going once warnings are fixed
   ```
   `docs/conf.py` calls `docs/gen_docs.py`, which writes `docs/_generated/*.md` (dataset/model tables from
   `DATASET_INFO` / `MODELS_INFO`) and copies the figure to `docs/_static/`. `.readthedocs.yaml` (Python 3.13, CPU
   torch, submodules) is untested on Read the Docs. `docs/_build/` was untracked from git.
2. **SARATRX on real data (#9), on the GPU.** Train on MSTAR/HRSID/SSDD with `freeze_encoder=True` and False, look
   at the anomaly maps, tune `learning_rate` (2.5e-4 default), epochs (20), `num_mask_passes` (4). Batch size/OOM was
   never exercised (the student reported OOM). The issue thread's other question ("zero-shot"): not possible, the
   released checkpoint has no decoder.
3. **Benchmark the new models** (YOLOAnomaly, MSFA, PadimACE, plus newer anomalib ones: Dinomaly, AnomalyDino,
   InpFormer, Glass, WinClip) on the real datasets with `SARIAD/config/default.yaml`; check whether MSFA's filters
   actually help.
4. **PaDiM-ACE whitening decision (needs the authors).** The reference code decomposes the *inverse* covariance
   and applies `D^-1/2 U^T`, which does not whiten (background variances 0.04-335 vs identity for the textbook
   `Sigma^-1/2`; the repo's bundled `MultiVariateGaussian` also stores the inverse). `whitening="reference"`
   (default) reproduces the original numerically (verified against a copy of its `ACELoss`); `"standard"` is textbook
   ACE. Decide which the paper's numbers correspond to and set the default; document in the paper's reproduction notes.
5. **Possible train/test overlap in MSTAR and SSDD** (pre-existing): the Folder test anomalies are `{split}/anom`, the
   same chips whose target-free versions (`{split}/norm`) are used for training. The new SAMPLE/SARDet datamodules use a
   disjoint test split. Worth a look before publishing results (noted in `docs/datasets.md`).
6. **Regenerate `figs/overall.svg`** (#19) to list the current components, if it should reflect the code.
7. **Rename `SARIAD/models/image/MFSA` -> `MSFA`** (upstream name; the issues use both spellings). The public class is
   already `MSFA`. Do the directory move on a native Linux filesystem or a fresh clone (see gotchas), and update
   `.gitmodules` and `SARIAD/models/__init__.py` (`_MODELS`).
8. **Issue #28's reference (IEEE 10433508) could not be retrieved** (empty page). MSFA follows the SARDet-100K code.
9. Nothing was pushed to PyPI and the version is unchanged (0.1.9). Open the PR, close issues via the merge, and cut a
   release if wanted.

## SARATRX findings (worth knowing before touching it)

- The checkpoint holds **only the encoder** (302 tensors). The decoder is random and must be trained.
- The original code built the MAE with the default config (`rpe=True`, 8 decoder blocks) instead of the one the
  checkpoint was trained with (`mae_hivit_base_dec512d6b`): 136 keys were silently randomly initialised.
- The MAE decoder predicts **per-patch-normalised multi-scale SAR gradient features** (`sarfeature1..3`, layout
  `(3, 16, 16)` per patch), not pixels. The old anomaly map compared the raw image to that output. The fix reuses the
  MAE's own target construction (`SARATRXModel.sar_features`) and averages the error over several random masks.
- anomalib's `Batch` **only accepts 3-channel images** (`anomalib/data/validators/torch/image.py`), which is what
  blocked the student ("Image must have 3 channels"). The pre-processor now emits 3 identical grayscale channels and the
  model reduces to one.

## Design decisions (so they are not redone)

- Feature-based models (YOLOAnomaly, MSFA) share `SARIAD/models/components/gaussian.py` (`FeatureGaussianModel` /
  `GaussianAD`, a `PadimModel`/`Padim` with a pluggable feature extractor).
- **`learning_type` must stay `ONE_CLASS`** for anything that needs the training pass: anomalib's `Engine.fit` skips
  training for `FEW_SHOT`/`ZERO_SHOT` models (this broke PaDiM-ACE until fixed).
- `SARIAD.models` is lazy (`_MODELS` in `SARIAD/models/__init__.py`) so importing it never needs a submodule; add new
  models there plus `MODELS_INFO`. Do not put anything except datamodules in `SARIAD.datasets.__all__` (the runner
  resolves names from it; there is a test).
- `normal_gen.remove_target` raises `SegmentationFailed` instead of writing a "normal" image that still contains the
  target: on dB-scaled SAMPLE chips the KMeans recipe selects the whole image. SAMPLE defaults to `qpm` scaling.
- Datamodule constructors no longer call `setup()` (it re-splits when not called by the Trainer).

## Environment gotchas (WSL on `/mnt/c`, the previous session's box)

- **Do not `git mv` a directory that contains an initialised submodule** on the NTFS mount: it failed halfway and WSL
  then showed two directories (`MFSA`, `MSFA`) with identical contents (stale cache) while Windows showed the truth.
  Use `cmd.exe /c dir` to see what NTFS really contains. It was cleaned up; only `MFSA` exists.
- Line endings: files were CRLF on the Windows checkout; `.gitattributes` (`* text=auto eol=lf`) normalises. If `git
  status` shows every file modified, run `git add --renormalize .`.
- `git` had **no user identity**; commits were made with `-c user.name=... -c user.email=...` per command (nothing was
  written to git config). Set your own.
- `pkill -f "<pattern>"` also matches the shell running it; use `pgrep -x`.
- `uv pip install -e ".[extra]"` stalled once on the mounted drive; installing the packages directly worked.
- Ultralytics downloads `*.pt` into the current directory; they are now git-ignored (as are `*.pth`, `*.ckpt`).
- The `SARATRX/SARATRX` submodule shows as dirty (` ?`) after imports because of `__pycache__`; harmless.
- Set `MLFLOW_DISABLE_AGENT_HINT=1` to silence a noisy hint from the mlflow that anomalib pulls in.

## Useful commands

```bash
sariad --config SARIAD/config/default.yaml --dry-run          # validate a config
python demo/train.py --dataset SSDD --model PadimACE --model-args '{"signature_dir": "path/to/anomalous/train/images"}'
pytest tests/test_models.py -k ace                             # PaDiM-ACE oracle/whitening tests
git log --oneline main..fix/open-issues                        # the commits, one per issue
```
