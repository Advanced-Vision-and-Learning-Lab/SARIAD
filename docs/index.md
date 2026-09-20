# SARIAD

**Benchmarking suite for synthetic aperture radar imagery anomaly detection (SARIAD) algorithms.**
SARIAD integrates SAR datasets, anomaly detection models and pre-processing methods with
[Anomalib](https://anomalib.readthedocs.io/) and evaluates them with a common set of metrics
([paper](https://arxiv.org/abs/2504.08115)).

<figure>
  <img src="_static/overall.svg" alt="Overall figure" style="max-width: 100%;">
  <figcaption>
    The flow of SARIAD, adapted from <a href="https://arxiv.org/abs/2202.08341">Anomalib</a>. The component lists
    in the figure date from the paper; see <a href="datasets.html">Datasets</a> and <a href="models.html">Models</a>
    for what is available now.
  </figcaption>
</figure>

## Install

```bash
git clone --recurse-submodules https://github.com/Advanced-Vision-and-Learning-Lab/SARIAD
cd SARIAD
pip install -e ".[dev]"          # Python >= 3.13
```

Some models need code from git submodules (SARATR-X, SAR-CNN); `--recurse-submodules` (or
`git submodule update --init`) fetches them. Importing SARIAD does not require them.

## Use

Describe experiments in a YAML file and run them, repeated and compared in a LaTeX table
(see `SARIAD/config/default.yaml`):

```bash
sariad --config SARIAD/config/default.yaml --dry-run    # check names and parameters, no downloads
sariad --config SARIAD/config/default.yaml
```

or use the pieces directly:

```python
from anomalib.engine import Engine
from SARIAD.datasets import SSDD
from SARIAD.models import PadimACE, YOLOAnomaly       # or any anomalib model, e.g. anomalib.models.Padim
from SARIAD.pre_processing import MedianFilter
from SARIAD.utils.inf import Inferencer

datamodule = SSDD()
model = YOLOAnomaly(weights="yolo11n.pt")
engine = Engine()
engine.fit(model=model, datamodule=datamodule)
print(Inferencer().evaluate(model, datamodule, engine=engine))
```

```{toctree}
:maxdepth: 2
:caption: Contents

datasets
models
source/modules
```
