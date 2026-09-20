import types

import pytest
import torch
from sklearn.metrics import roc_auc_score

from SARIAD.utils import inf


def make_batch(n=8, size=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    gt_label = torch.rand(n, generator=g) > 0.5
    gt_mask = torch.zeros(n, size, size, dtype=torch.bool)
    gt_mask[gt_label, 4:10, 4:10] = True
    amap = torch.rand(n, 1, size, size, generator=g)
    amap[gt_label, :, 4:10, 4:10] += 0.35
    score = amap.amax((-2, -1)).reshape(n) + 0.15 * torch.randn(n, generator=g)
    return types.SimpleNamespace(gt_label=gt_label, pred_label=score > 0.9, pred_score=score, gt_mask=gt_mask,
                                 pred_mask=amap[:, 0] > 0.8, anomaly_map=amap)


@pytest.fixture
def predictions():
    return [make_batch(seed=s) for s in range(5)]


def test_pixel_auroc_uses_the_continuous_anomaly_map(predictions):
    metrics = inf.Metrics(predictions).get_all_metrics()
    y = torch.cat([b.gt_mask.reshape(-1) for b in predictions]).numpy()
    s = torch.cat([b.anomaly_map.reshape(-1) for b in predictions]).numpy()
    assert metrics["AUROC (Pixel-level)"] == pytest.approx(roc_auc_score(y, s), abs=1e-6)


def test_image_metrics_are_consistent(predictions):
    m = inf.Metrics(predictions).get_all_metrics()
    assert m["TP"] + m["TN"] + m["FP"] + m["FN"] == 40
    assert m["Accuracy"] == pytest.approx((m["TP"] + m["TN"]) / 40)
    assert m["Missed Alarm Rate (MAR)"] == pytest.approx(1 - m["Recall/Sensitivity"])
    y = torch.cat([b.gt_label for b in predictions]).numpy()
    s = torch.cat([b.pred_score for b in predictions]).numpy()
    assert m["AUROC (Image-level)"] == pytest.approx(roc_auc_score(y, s), abs=1e-6)


def test_streaming_equals_single_batch(predictions):
    big = types.SimpleNamespace(**{k: torch.cat([getattr(b, k) for b in predictions]) for k in vars(predictions[0])})
    streamed, once = inf.Inferencer(), inf.Inferencer()
    for batch in predictions:
        streamed.update(batch)
    once.update(big)
    a, b = streamed.compute(), once.compute()
    assert all(a[k] == pytest.approx(b[k], abs=1e-6) for k in a)


def test_binned_pixel_auroc_is_close(predictions):
    exact = inf.Inferencer.from_predictions(predictions).compute()["AUROC (Pixel-level)"]
    binned = inf.Inferencer.from_predictions(predictions, pixel_thresholds=500).compute()["AUROC (Pixel-level)"]
    assert binned == pytest.approx(exact, abs=5e-3)


def test_reset(predictions):
    inferencer = inf.Inferencer.from_predictions(predictions)
    inferencer.reset()
    inferencer.update(predictions[0])
    assert sum(inferencer.compute()[k] for k in ("TP", "TN", "FP", "FN")) == 8


def test_missing_masks_and_single_class_are_reported_not_faked(predictions):
    no_masks = types.SimpleNamespace(**{**vars(predictions[0]), "gt_mask": None, "pred_mask": None, "anomaly_map": None})
    result = inf.Inferencer.from_predictions([no_masks]).compute()
    assert "N/A" in result["AUROC (Pixel-level)"] and isinstance(result["AUROC (Image-level)"], float)

    one_class = types.SimpleNamespace(**{**vars(predictions[0]), "gt_label": torch.ones(8, dtype=torch.bool)})
    assert "N/A" in inf.Inferencer.from_predictions([one_class]).compute()["AUROC (Image-level)"]


def test_metrics_outputs(predictions, tmp_path):
    metrics = inf.Metrics(predictions)
    metrics.save_all(str(tmp_path / "nested" / "out"))   # creates parents
    assert {p.name for p in (tmp_path / "nested" / "out").iterdir()} >= {"metrics.txt", "metrics_table.tex", "roc_curve.png"}
    table = inf.Metrics.compare_multiple_runs({"a": [metrics, metrics], "b": metrics}, output_dir=str(tmp_path / "cmp"))
    assert (tmp_path / "cmp" / "roc_curve_comparison.png").exists() and r"\pm" in table and r"\textbf" in table


def test_invalid_metric_name():
    with pytest.raises(ValueError, match="Invalid metrics"):
        inf.Metrics(None, ["Nope"])
