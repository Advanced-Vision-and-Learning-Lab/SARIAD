"""Tests that need downloads or checkpoints. Off by default so CI stays offline:

    SARIAD_TEST_NETWORK=1 pytest tests/test_models_optional.py                # YOLO weights are downloaded
    SARIAD_TEST_SARATRX_CHECKPOINT=/path/mae_hivit_base_1600ep.pth pytest ... # needs the SARATR-X submodule too
"""

import os
from pathlib import Path

import pytest
import torch

network = pytest.mark.skipif(not os.environ.get("SARIAD_TEST_NETWORK"), reason="set SARIAD_TEST_NETWORK=1 to download weights")
SARATRX_CHECKPOINT = os.environ.get("SARIAD_TEST_SARATRX_CHECKPOINT")
saratrx = pytest.mark.skipif(
    not (SARATRX_CHECKPOINT and Path(SARATRX_CHECKPOINT).is_file()
         and (Path(__file__).parents[1] / "SARIAD/models/image/SARATRX/SARATRX/pretraining").is_dir()),
    reason="needs SARIAD_TEST_SARATRX_CHECKPOINT and the SARATR-X submodule",
)


@network
@pytest.mark.parametrize("weights", ["yolov8n.pt", "yolo11n.pt"])
def test_yolo_anomaly_model(weights):
    from SARIAD.models.image.YOLO.torch_model import YOLOAnomalyModel

    model = YOLOAnomalyModel(weights, n_features=50)
    assert len(model.feature_extractor.out_dims) == 3
    x = torch.rand(6, 3, 256, 256)
    model.train()
    model(x)
    model.fit()
    model.eval()
    assert model(x[:2]).anomaly_map.shape == (2, 1, 256, 256)
    assert all(not p.requires_grad for p in model.feature_extractor.parameters())


@network
def test_yolo_rejects_a_missing_layer():
    from SARIAD.models.image.YOLO.torch_model import YOLOAnomalyModel

    with pytest.raises(ValueError, match="does not exist"):
        YOLOAnomalyModel("yolo11n.pt", layers=(4, 6, 999))


@saratrx
def test_saratrx_error_equals_the_mae_training_loss_for_a_fixed_mask():
    from SARIAD.models.image.SARATRX.torch_model import SARATRXModel

    net = SARATRXModel(checkpoint_path=SARATRX_CHECKPOINT).eval()
    x = torch.rand(2, 1, 224, 224) + 0.1
    with torch.no_grad():
        target = net.sar_features(x)
        torch.manual_seed(0)
        loss, pred, mask = net.model(x, 0.75)
        error = net.anomaly_map_generator.patch_error(pred, target).mean(-1)
    assert float((error * mask).sum() / mask.sum()) == pytest.approx(loss.item(), abs=1e-5)
    out = net(x)
    assert out.anomaly_map.shape == (2, 1, 224, 224) and out.pred_score.shape == (2,)
    with pytest.raises(ValueError, match="224"):
        net(torch.rand(1, 1, 128, 128))


@saratrx
def test_saratrx_checkpoint_has_no_missing_encoder_weights_and_freezing_works():
    from SARIAD.models.image.SARATRX.torch_model import SARATRXModel

    full = SARATRXModel(checkpoint_path=SARATRX_CHECKPOINT)
    frozen = SARATRXModel(checkpoint_path=SARATRX_CHECKPOINT, freeze_encoder=True)
    count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert count(frozen) < count(full) / 3
