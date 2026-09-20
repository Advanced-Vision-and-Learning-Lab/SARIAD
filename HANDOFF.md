# Handoff: SARIAD after the "open issues" PR

Written for the next Claude Code session (or person). This file lives on the `fix/open-issues` branch on purpose and
is **not** on `main`.

- **PR #47 is merged** into `main` as a squash commit, `9905e97`
  (https://github.com/Advanced-Vision-and-Learning-Lab/SARIAD/pull/47). `fix/open-issues` still exists on `origin` with
  the 20 original commits; its tree is identical to `main`'s, only the history differs. **Start follow-up work from a
  fresh branch off `main`**, not from this one.
- **Reference paper:** https://arxiv.org/abs/2504.08115 (SARIAD). PaDiM-ACE: https://arxiv.org/abs/2504.08049.
- The first session (CPU-only WSL box) wrote the code; the second (Windows 11, RTX 2050 4 GB) ran it for the first
  time, fixed the docs build and CI, and did the merge. Nothing has been run on real SAR data yet.

## Where the issues stand

| Issue | State | Notes |
|---|---|---|
| #2 #4 #5 #6 #7 #8 #14 #15 #16 #17 #27 #32 #43 | closed by the merge | |
| #9 SARATR-X | **open, on purpose** | works on GPU with the real checkpoint; no real-data result yet |
| #28 / #39 MSFA | **open, on purpose** | only the data-input stage of the SARDet-100K method is implemented |
| #19 overall figure | **closed, but the figure is stale** | closed from the web UI by `jpeeples67` right after the merge, not by the merge; `figs/overall.svg` still lists only "SARATR-X (WIP)" etc. |

Comments explaining the state were posted on #9, #28, #39 and #19.

**Lesson from the merge:** GitHub's default squash message concatenates every commit message, so the `Fixes #9`,
`#28` and `#39` lines closed those three issues even though the intended message left them out. They were reopened by
hand. When squash-merging, replace the whole commit body, or use `Refs #N` in commits for issues that must stay open.

## Getting set up

Windows / PowerShell (the second session's box):

```powershell
git fetch; git checkout -b my-branch origin/main
git submodule update --init                      # NOT --recursive, see gotchas
uv venv --python 3.13
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv\Scripts\python.exe -e ".[dev,docs]"
$env:MLFLOW_DISABLE_AGENT_HINT='1'; .venv\Scripts\python.exe -m pytest
```

On Linux the same steps apply with `source .venv/bin/activate`.

The pretrained SARATR-X checkpoint (`mae_hivit_base_1600ep.pth`, 263 MB, Google Drive id in
`SARIAD/models/image/SARATRX/torch_model.py`) downloads into anomalib's pretrained-weights cache on first use
(`%LOCALAPPDATA%\anomalib\anomalib\Cache\pre_trained\` on Windows). To run the gated tests set
`SARIAD_TEST_SARATRX_CHECKPOINT=<that path>` and `SARIAD_TEST_NETWORK=1`.

## What was and was not verified

**Verified (Windows, RTX 2050 4 GB, torch 2.14+cu126, anomalib >= 2.6.2):**
- `pytest`: 76 tests. 71 run offline and pass; the 5 gated ones (YOLO weights download, SARATRX real checkpoint)
  are skipped by default and **also pass** with the two environment variables set. The SARATRX test checks that the
  anomaly error equals the MAE's own training loss for a fixed mask.
  `tests/test_pipeline.py::test_runner_end_to_end` was **intermittently failing** on this Windows box (about 2 of 5
  full runs) with `_tkinter.TclError: Can't find a usable init.tcl`: matplotlib defaulted to the Tk backend and Tk
  init failed at random. `tests/conftest.py` now sets `MPLBACKEND=Agg`; 4 consecutive full runs passed afterwards
  (a small sample, so keep an eye on it).
- `sariad --config SARIAD/config/default.yaml --dry-run`: "Config OK: 3 experiment(s) validated".
- `sphinx-build -W --keep-going -b html docs docs/_build/html` builds with **zero warnings**.
- CI (`Test`, `Build distribution`) is green on the last commit of the PR (`a3c96e0`).
- SARATRX on the GPU with the real checkpoint, one train step and one inference pass (3 identical grayscale channels
  in, 224x224). Peak memory / time:

  | encoder | batch | train | inference |
  |---|---|---|---|
  | frozen | 1 / 4 / 8 / 16 | 0.67 / 0.88 / 1.15 / 1.71 GiB | 0.60-0.75 GiB |
  | finetuned | 1 / 4 / 8 / 16 | 1.67 / 1.86 / 2.35 / 3.40 GiB | 1.35-1.51 GiB |

  Nothing ran out of memory up to batch 16, so the OOM the student reported in #9 does not reproduce on 4 GB with
  these settings. Speed is roughly 0.7 s per training step and 0.9 s per inference batch at batch 16.
  (Numbers use random inputs: they measure memory and speed, not anomaly quality.)

**Still not run:**
- **Any result on real SAR data.** No dataset exists on either machine. No claim is made about SARATRX/YOLO/MSFA/ACE
  quality on MSTAR/HRSID/SSDD; synthetic AUROCs only show wiring.
- The full SAMPLE download (1.5 GB) and the SARDet-100K Kaggle download (tens of GB, needs credentials). SARDet
  generation was tested on a synthetic COCO fixture only, and its patch inpainting may be slow on 800x800 images.
- The BiRefNet segmentation backend (`normal_gen.segment_birefnet`): never executed (1 GB download,
  `trust_remote_code=True`). Treat as experimental.
- `.readthedocs.yaml` on Read the Docs itself (Python 3.13, CPU torch, submodules). The docs build locally.
- No PyPI release. The version is unchanged (0.1.9); the publish job only runs on tags.

## Open items, in priority order

1. **SARATRX on real data (#9).** Train on MSTAR/HRSID/SSDD with `freeze_encoder=True` and `False`, look at the anomaly
   maps, tune `learning_rate` (2.5e-4 default), epochs (20), `num_mask_passes` (4). Memory is not the constraint (see
   above). The "zero-shot" question in the thread: not possible, the released checkpoint has no decoder. Close #9
   once there is a real result.
2. **Benchmark the new models** (YOLOAnomaly, MSFA, PadimACE, plus newer anomalib ones: Dinomaly, AnomalyDino,
   InpFormer, Glass, WinClip) on the real datasets with `SARIAD/config/default.yaml`; check whether MSFA's filters
   actually help. That result decides whether #28/#39 can close or need the rest of the SARDet-100K method.
3. **PaDiM-ACE whitening decision (needs the authors).** The reference code decomposes the *inverse* covariance and
   applies `D^-1/2 U^T`, which does not whiten (background variances 0.04-335 vs identity for the textbook
   `Sigma^-1/2`; the repo's bundled `MultiVariateGaussian` also stores the inverse). `whitening="reference"` (default)
   reproduces the original numerically (verified against a copy of its `ACELoss`); `"standard"` is textbook ACE.
   Decide which the paper's numbers correspond to, set the default, and document it in the paper's reproduction notes.
4. **Possible train/test overlap in MSTAR and SSDD** (pre-existing): the Folder test anomalies are `{split}/anom`, the
   same chips whose target-free versions (`{split}/norm`) are used for training. The SAMPLE/SARDet datamodules use a
   disjoint test split. Check before publishing results (noted in `docs/datasets.md`).
5. **Regenerate `figs/overall.svg`** to list the current components. Its text is outlined paths, so it cannot be
   edited in place. #19 is closed but this is not done; reopen it or open a new issue.
6. **Rename `SARIAD/models/image/MFSA` -> `MSFA`** (upstream name; the issues use both spellings). The public class is
   already `MSFA`. Do the directory move in a fresh clone or on a native filesystem (see gotchas), and update
   `.gitmodules` and `SARIAD/models/__init__.py` (`_MODELS`).
7. **Issue #28's reference (IEEE 10433508) could not be retrieved** (empty page). MSFA follows the SARDet-100K code.
8. **Library plots still use pyplot's default (GUI) backend.** `SARIAD/utils/inf.py` only saves figures
   (`savefig` + `close`), but on Windows/desktop Linux it goes through the Tk backend, so a broken Tk install fails a
   real run the same way the test did (the fix so far is only in `tests/conftest.py`). Cleaner: build the figures with
   `matplotlib.figure.Figure` (no backend involved) instead of `plt.*`. `img_utils.py` calls `plt.show()` on purpose.
9. **CI housekeeping:** GitHub warns that `actions/checkout@v4` runs on a deprecated Node 20, and that `ubuntu-latest`
   moves to Ubuntu 26 on 2026-10-19. Neither fails today.
10. Cut a release if wanted (bump the version and push a tag).

## SARATRX findings (worth knowing before touching it)

- The checkpoint holds **only the encoder** (302 tensors). The decoder is random and must be trained.
- The original code built the MAE with the default config (`rpe=True`, 8 decoder blocks) instead of the one the
  checkpoint was trained with (`mae_hivit_base_dec512d6b`): 136 keys were silently randomly initialised.
- The MAE decoder predicts **per-patch-normalised multi-scale SAR gradient features** (`sarfeature1..3`, layout
  `(3, 16, 16)` per patch), not pixels. The old anomaly map compared the raw image to that output. The fix reuses the
  MAE's own target construction (`SARATRXModel.sar_features`) and averages the error over several random masks.
- anomalib's `Batch` **only accepts 3-channel images** (`anomalib/data/validators/torch/image.py`), which is what
  blocked the student ("Image must have 3 channels"). The pre-processor emits 3 identical grayscale channels and the
  model reduces to one. Inputs must be 224x224.

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
- The docs tables come from `DATASET_INFO` / `MODELS_INFO` via `docs/gen_docs.py` (run by `docs/conf.py`); keep the
  metadata next to the code. `autodoc_inherit_docstrings = False` is deliberate (see gotchas).

## Environment gotchas

**CI and git**
- **Never use `submodules: recursive`, or even `submodules: true`, in `actions/checkout`.** The SARATR-X fork has a
  gitlink `detection/apex` with no `.gitmodules`, and `actions/checkout` always runs `git submodule foreach --recursive`
  to manage credentials, which fails ("No url found for submodule path ... detection/apex"). The workflow initialises
  the submodules in a plain `git submodule update --init` step instead. The same applies locally: do not pass
  `--recursive`.
- CI logs cannot be downloaded without auth (HTTP 403), but the failing step and annotations are readable through the
  public API: `.../actions/jobs/<id>` and `.../check-runs/<id>/annotations`.
- `git` has **no user identity** configured; commits were made with `-c user.name=... -c user.email=...` per command
  (nothing was written to git config). Set your own.
- Line endings: `.gitattributes` (`* text=auto eol=lf`) normalises. If `git status` shows every file modified, run
  `git add --renormalize .`. The SARATRX submodule shows as dirty (` ?`) after imports because of `__pycache__`;
  harmless.

**Windows session**
- In Claude Code, a `!` command runs in **Git Bash**, not PowerShell: `winget` and `~/.local/bin` are not on its PATH.
  `winget` is not installed on this machine at all.
- `gh` v2.101.0 was installed as a portable binary in `C:\Users\jpeeples\.local\bin\gh.exe` (checksum verified) and
  signed in as `jpeeples67` over SSH. Run `gh auth logout` when done. In PowerShell 5.1, passing a multi-line string
  that contains double quotes to a native command splits the argument; use `--body-file`.
- Redirecting a native command's stderr (`2>&1`) in PowerShell 5.1 turns git's progress output into red
  "NativeCommandError" text even on success; check the exit code instead.
- `pyproject.toml` sets `addopts = "-q"`, so `pytest -q` is doubly quiet and prints no summary line. Use `-rs` or `-v`
  to see counts and skip reasons.
- `docs/conf.py` sets `autodoc_inherit_docstrings = False`: `Transform.transform` inherits a torchvision docstring that
  links to a label only torchvision's docs define.

**Previous (WSL on `/mnt/c`) session**
- **Do not `git mv` a directory that contains an initialised submodule** on the NTFS mount from WSL: it failed halfway
  and WSL then showed two directories (`MFSA`, `MSFA`) with identical contents (stale cache). Use `cmd.exe /c dir` to see
  what NTFS really contains. Only `MFSA` exists.
- `pkill -f "<pattern>"` also matches the shell running it; use `pgrep -x`.
- `uv pip install -e ".[extra]"` stalled once on the mounted drive; installing the packages directly worked.
- Ultralytics downloads `*.pt` into the current directory; they are git-ignored (as are `*.pth`, `*.ckpt`).
- Set `MLFLOW_DISABLE_AGENT_HINT=1` to silence a noisy hint from the mlflow that anomalib pulls in.

## Useful commands

```powershell
sariad --config SARIAD/config/default.yaml --dry-run          # validate a config
python demo/train.py --dataset SSDD --model PadimACE --model-args '{"signature_dir": "path/to/anomalous/train/images"}'
pytest tests/test_models.py -k ace                             # PaDiM-ACE oracle/whitening tests
sphinx-build -W --keep-going -b html docs docs/_build/html     # must stay warning-free
sphinx-apidoc -f -o docs/source SARIAD SARIAD/models/image/SARATRX/SARATRX SARIAD/pre_processing/SARCNN/SARCNN_SRC SARIAD/models/image/MFSA/SARDet_100K SARIAD/results   # after adding modules
```
