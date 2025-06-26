# Benchmarking suite for synthetic aperture radar imagery anomaly detection (SARIAD) algorithms

## Overview
This package is designed for anomaly detection in Synthetic Aperture Radar (SAR) images, leveraging PyTorch Lightning and models from [Anomalib](https://anomalib.readthedocs.io/). The package is modular, allowing easy benchmarking and dataset integration.

## Directory Structure
```
SARIAD/
├── config/
│   ├── default.yaml
│   ├── environment.yaml  # Conda environment file
├── datasets/
│   ├── __init__.py
│   ├── MSTAR/
│   ├── custom_dataset/
│   └── sar_datamodule.py  
├── models/
│   ├── __init__.py
│   ├── anomalib_models.py
│   ├── autoencoder.py
│   └── transformer.py
├── preprocessing/
│   ├── __init__.py
│   ├── normalize.py
│   ├── augmentations.py
│   └── utils.py
├── benchmarks/
│   ├── __init__.py
│   └── benchmarking.py
├── main.py
├── __init__.py
└── utils/
    ├── __init__.py
    └── config_loader.py
```

## Installation
Our package is on PyPI and thus can simply be installed with `pip install SARIAD` using `python>=3.13`.

### Development Installation
```bash
# Clone the repository
git clone https://github.com/Advanced-Vision-and-Learning-Lab/SARIAD

# Install SARIAD in editable mode
pip install -e .
```

## Configuration
Edit the YAML file located in `config/default.yaml` to specify the dataset path, model, and training parameters.

## Usage
```bash
# Train with a specific configuration
python main.py --config config/default.yaml
```

## Benchmarking
To enable benchmarking, set `benchmark.enabled: True` in the YAML file and specify the number of runs.

## Preprocessing
- **normalize.py**: Functions for data normalization.
- **augmentations.py**: Functions for data augmentations.
- **utils.py**: Utility functions for SAR-specific preprocessing.

The `SARDataModule` located in the `datasets` folder imports these functions to ensure consistent preprocessing across datasets.

## License
MIT License

## Acknowledgments
This project is inspired by [Anomalib](https://anomalib.readthedocs.io/) and [Benchmarks for Medical Anomaly Detection (BMAD)](https://github.com/dorisbao/bmad).

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository on GitHub.
2. Create a new branch with a descriptive name.
3. Make your changes and ensure they follow the code style guidelines.
4. Write unit tests for any new features or bug fixes.
5. Submit a pull request with a clear description of your changes.

For major changes, please open an issue first to discuss what you'd like to change. We appreciate your contributions to improve this work!

## Citing SARIAD

If you use the SARIAD code, please cite the following reference using the following entry.

**Plain Text:**

L. Chauvin, S. Gupta, A. Ibarra and J. Peeples, "Benchmarking suite for synthetic aperture radar imagery anomaly detection (SARIAD) algorithms," in Algorithms for Synthetic Aperture Radar Imagery XXXII, vol. TBD. International Society for Optics and Photonics (SPIE), 2025, [DOI: 10.1117/12.3052519](https://doi.org/10.1117/12.3052519)

[![arXiv](http://img.shields.io/badge/cs.CV-arXiv%3A2504.08115-B31B1B.svg)](https://doi.org/10.48550/arXiv.2504.08115)

**BibTex:**

```
@inproceedings{Chauvin2025Benchmarking,
  title={Benchmarking suite for synthetic aperture radar imagery anomaly detection (SARIAD) algorithms},
  author={Chauvin, Lucian and Gupta, Somil, and Ibarra, Angelina, and Peeples, Joshua},
  booktitle={Algorithms for Synthetic Aperture Radar Imagery XXXII},
  pages={TBD},
  year={2025},
  organization={International Society for Optics and Photonics (SPIE)}
  doi={10.1117/12.3052519}
}
```

## Citing MSTAR
If you use this dataset in your research, please cite the following paper:
```
@misc{mstar2025,
  title = {MSTAR Public Dataset},
  author = {{U.S. Air Force}},
  year = {1995},
  note = {Sensor Data Management System (SDMS)},
  url = {https://www.sdms.afrl.af.mil/index.php?collection=mstar}
}
```
