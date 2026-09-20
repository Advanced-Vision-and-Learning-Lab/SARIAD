import pytest

from SARIAD.config import run


def test_settings_top_level_and_legacy_global():
    assert run._settings({"seed": 1, "output_dir": "a"}) == {"seed": 1, "output_dir": "a"}
    assert run._settings({"seed": 1, "global": {"seed": 2}})["seed"] == 2


@pytest.mark.parametrize("name", ["MSTAR", "mstar", "SSDD", "SARDet_100K", "sample_public"])
def test_resolve_dataset(name):
    assert run.resolve_dataset(name).__name__.lower() == name.lower()


def test_resolve_dataset_unknown():
    with pytest.raises(ValueError, match="Unknown dataset"):
        run.resolve_dataset("nope")


@pytest.mark.parametrize("name", ["Padim", "EfficientAd", "Patchcore", "PadimACE", "YOLOAnomaly", "MSFA"])
def test_resolve_model(name):
    assert run.resolve_model(name).__name__ == name


def test_resolve_model_unknown():
    with pytest.raises(ValueError, match="Unknown model"):
        run.resolve_model("NotAModel")


def test_validate_config_accepts_shipped_configs():
    import yaml
    from pathlib import Path

    root = Path(__file__).parents[1]
    for path in (root / "SARIAD/config/default.yaml", root / "demo/demo.yaml"):
        assert run.validate_config(yaml.safe_load(path.read_text())) == []


def test_validate_config_reports_every_problem():
    config = {
        "experiments": [
            {"name": "a", "dataset": "MSTAR", "preprocessor": "NLM", "model": "Padim"},
            {"name": "b", "dataset": "NoSuchSet", "model": "Padim"},
        ],
        "datasets": {"MSTAR": {"bogus": 1}},
        "models": {"Padim": {"input_size": [256, 256]}},
        "preprocessors": {"NLM": {"h": 10}},
    }
    problems = "\n".join(run.validate_config(config))
    assert "'bogus'" in problems and "'input_size'" in problems and "'h'" in problems and "NoSuchSet" in problems


def test_validate_config_accepts_folder_kwargs_through_datamodule():
    config = {"experiments": [{"name": "a", "dataset": "SSDD", "model": "Padim"}], "datasets": {"SSDD": {"batch_size": 4, "seed": 1, "num_workers": 0}}}
    assert run.validate_config(config) == []


def test_main_dry_run(tmp_path):
    good = tmp_path / "good.yaml"
    good.write_text("experiments:\n  - {name: a, dataset: MSTAR, model: Padim}\n")
    bad = tmp_path / "bad.yaml"
    bad.write_text("experiments:\n  - {name: a, dataset: MSTAR, model: Padim}\nmodels: {Padim: {nope: 1}}\n")
    assert run.main(["-c", str(good), "--dry-run"]) == 0
    assert run.main(["-c", str(bad), "--dry-run"]) == 1
    assert run.main(["-c", str(tmp_path / "missing.yaml")]) == 1
