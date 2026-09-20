"""End to end: config -> datamodule -> model -> Engine -> metrics, on a synthetic dataset, offline and on CPU."""

import pytest

import SARIAD.datasets as sariad_datasets
from SARIAD.config.run import run_experiments


@pytest.fixture
def synth_registered(synthetic_datamodule):
    class Synth(type(synthetic_datamodule())):
        def __init__(self, batch_size=8, num_workers=0, **kwargs):
            base = synthetic_datamodule(batch_size, num_workers, **kwargs)
            self.__dict__.update(base.__dict__)                                       # reuse the configured Folder
            self.__class__ = Synth

    sariad_datasets.Synth = Synth
    sariad_datasets.__all__.append("Synth")
    yield
    sariad_datasets.__all__.remove("Synth")
    del sariad_datasets.Synth


def test_runner_end_to_end(tmp_path, synth_registered):
    config = {
        "seed": 1,
        "benchmark_runs": 2,
        "output_dir": str(tmp_path / "results"),
        "experiments": [{"name": "padim", "dataset": "Synth", "model": "Padim"}],
        "datasets": {"Synth": {"batch_size": 8}},
        "models": {"Padim": {"backbone": "resnet18", "layers": ["layer1"], "n_features": 30, "pre_trained": False}},
    }
    result = run_experiments(config)
    assert len(result["padim"]) == 2                                                  # benchmark_runs honored
    out = tmp_path / "results"
    assert (out / "comparison_table.tex").exists() and (out / "roc_curve_comparison.png").exists()
    assert (out / "padim" / "run_0" / "metrics.txt").exists() and (out / "padim" / "run_1" / "metrics.txt").exists()
    metrics = result["padim"][0].get_all_metrics()
    assert 0.0 <= metrics["AUROC (Image-level)"] <= 1.0 and 0.0 <= metrics["AUROC (Pixel-level)"] <= 1.0


def test_runner_fails_loudly_when_nothing_succeeds(tmp_path, synth_registered):
    config = {"output_dir": str(tmp_path), "experiments": [{"name": "x", "dataset": "Synth", "model": "Padim", "runs": 1}],
              "models": {"Padim": {"backbone": "no_such_backbone", "pre_trained": False}}}
    with pytest.raises(RuntimeError, match="No experiment completed"):
        run_experiments(config)


def test_padim_ace_pipeline(tmp_path, synth_registered, synthetic_folder):
    config = {
        "output_dir": str(tmp_path / "results"),
        "experiments": [{"name": "ace", "dataset": "Synth", "model": "PadimACE", "runs": 1}],
        "models": {"PadimACE": {"backbone": "resnet18", "layers": ["layer1"], "n_features": 30, "pre_trained": False,
                                "signature_dir": str(synthetic_folder / "signatures"), "cov_type": "diagonal"}},
    }
    metrics = run_experiments(config)["ace"][0].get_all_metrics()
    assert metrics["AUROC (Pixel-level)"] > 0.5                                       # the signatures localize the synthetic anomalies


def test_msfa_pipeline(tmp_path, synth_registered):
    config = {"output_dir": str(tmp_path / "results"), "experiments": [{"name": "msfa", "dataset": "Synth", "model": "MSFA", "runs": 1}],
              "models": {"MSFA": {"pre_trained": False, "n_features": 30}}}
    assert "msfa" in run_experiments(config)
