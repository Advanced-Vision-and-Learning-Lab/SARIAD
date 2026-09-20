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
