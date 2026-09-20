import pathlib

import pytest
import torch
from torchvision import tv_tensors


def test_pre_processing_imports_without_the_sarcnn_submodule():
    """Regression: SAR-CNN weights used to be loaded at import time, breaking every preprocessor on CPU-only machines."""
    import SARIAD.pre_processing as pp

    assert {"SARCNN", "NLM", "MedianFilter", "Default"} <= set(dir(pp))


def test_sarcnn_gives_a_clear_error_without_weights(monkeypatch):
    import sys

    import SARIAD.pre_processing  # noqa: F401  (registers the module)

    module = sys.modules["SARIAD.pre_processing.SARCNN.SARCNN"]
    monkeypatch.setattr(module, "WEIGHTS_DIR", pathlib.Path("/nonexistent"))
    module.load_sarcnn_net.cache_clear()
    with pytest.raises(RuntimeError, match="submodule"):
        module.load_sarcnn_net()
    module.load_sarcnn_net.cache_clear()


def test_sarcnn_log_domain_roundtrip_is_finite_on_zero_pixels():
    import sys

    import SARIAD.pre_processing  # noqa: F401

    module = sys.modules["SARIAD.pre_processing.SARCNN.SARCNN"]
    x = torch.zeros(1, 1, 8, 8)
    assert torch.isfinite(module.preprocessing_int2net(x)).all()
    y = torch.rand(1, 1, 8, 8) + 0.1
    assert torch.allclose(module.postprocessing_net2int(2 * module.preprocessing_int2net(y) / 2), y, atol=1e-5)


def test_median_filter_transform_runs_on_cpu():
    from anomalib.models import Padim

    from SARIAD.pre_processing import MedianFilter

    pre = MedianFilter(model=Padim, use_cuda=False)
    batch = tv_tensors.Image(torch.rand(2, 3, 64, 64))
    out = pre.transform(batch)
    assert out.shape[0] == 2 and out.shape[1] == 3
