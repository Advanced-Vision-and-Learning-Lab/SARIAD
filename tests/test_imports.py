import importlib

import pytest


@pytest.mark.parametrize("module", ["SARIAD", "SARIAD.models", "SARIAD.datasets", "SARIAD.pre_processing", "SARIAD.utils.inf",
                                    "SARIAD.utils.normal_gen", "SARIAD.config.run"])
def test_imports(module):
    """Importing must not need a GPU, downloads, or the git submodules."""
    importlib.import_module(module)


def test_models_are_lazy_and_exported():
    import SARIAD.models as models

    assert {"SARATRX", "YOLOAnomaly", "PadimACE", "MSFA"} <= set(models.__all__)
    with pytest.raises(AttributeError):
        models.DoesNotExist  # noqa: B018
    # lightweight ones import without any third-party checkout
    assert models.PadimACE.__name__ == "PadimACE" and models.MSFA.__name__ == "MSFA"


def test_datasets_exported():
    import SARIAD.datasets as datasets

    assert {"MSTAR", "HRSID", "SSDD", "SAMPLE_PUBLIC", "SARDet_100K"} <= set(datasets.__all__)


def test_datasets_all_lists_only_datamodules_and_info_covers_them():
    """The runner resolves dataset names from __all__: metadata must not leak into it."""
    import SARIAD.datasets as datasets

    assert "DATASETS_INFO" not in datasets.__all__
    assert set(datasets.DATASETS_INFO) == set(datasets.__all__)
    for info in datasets.DATASETS_INFO.values():
        assert {"name", "summary", "source", "download", "anomaly", "normal_data", "masks"} <= set(info)


def test_models_info_covers_exported_models():
    import SARIAD.models as models

    assert set(models.MODELS_INFO) == set(models.__all__)
    assert all({"summary", "paper", "code", "training"} <= set(info) for info in models.MODELS_INFO.values())
