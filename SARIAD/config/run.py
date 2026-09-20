"""Run SARIAD experiments described by a YAML file.

The YAML file lists ``experiments`` (dataset + model + optional preprocessor) and the
constructor parameters of each dataset/model/preprocessor in ``datasets``, ``models`` and
``preprocessors``. See ``SARIAD/config/default.yaml`` for an annotated example.
"""

import argparse
import importlib
import inspect
import logging
import pickle
from pathlib import Path

import torch
import yaml
from lightning.pytorch import seed_everything
from tqdm import tqdm

from anomalib.engine import Engine
from SARIAD.utils import inf

logger = logging.getLogger(__name__)


def _settings(config: dict) -> dict:
    """Global settings (``seed``, ``output_dir``, ``benchmark_runs``).

    They live at the top level of the YAML file. A nested ``global:`` block is still
    accepted and takes precedence, for configs written for older versions.
    """
    return {**{k: v for k, v in config.items() if k != "global"}, **config.get("global", {})}


def resolve_dataset(name: str):
    """Return the datamodule class called ``name`` from :mod:`SARIAD.datasets` (case-insensitive)."""
    datasets = importlib.import_module("SARIAD.datasets")
    available = {n.lower(): n for n in getattr(datasets, "__all__", dir(datasets)) if not n.startswith("_")}
    if name.lower() not in available:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(available.values())}")
    return getattr(datasets, available[name.lower()])


def resolve_model(name: str):
    """Return the model class called ``name``.

    SARIAD models (``SARIAD.models``) take precedence, then any Anomalib model
    (``anomalib.models``, e.g. ``Padim``, ``EfficientAd``, ``Dinomaly``).
    """
    sariad_models = importlib.import_module("SARIAD.models")
    if name in getattr(sariad_models, "__all__", ()):
        return getattr(sariad_models, name)

    anomalib_models = importlib.import_module("anomalib.models")
    if hasattr(anomalib_models, name):
        return getattr(anomalib_models, name)

    # Fall back to the module layout (anomalib.models.image.<snake_case_name>)
    from anomalib.utils.path import convert_to_snake_case

    try:
        module = importlib.import_module(f"anomalib.models.image.{convert_to_snake_case(name)}")
        return getattr(module, name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Unknown model '{name}': not in SARIAD.models or anomalib.models") from e


def resolve_preprocessor(name: str):
    """Return the preprocessor class called ``name`` from :mod:`SARIAD.pre_processing`."""
    module = importlib.import_module("SARIAD.pre_processing")
    if not hasattr(module, name):
        raise ValueError(f"Unknown preprocessor '{name}'")
    return getattr(module, name)


def _check_params(kind: str, name: str, cls, params: dict, injected: tuple[str, ...] = ()) -> list[str]:
    """Return the problems (if any) binding ``params`` to ``cls.__init__``."""
    # Collect the accepted names, following **kwargs up the class hierarchy (e.g. a datamodule
    # forwarding **folder_kwargs to anomalib's Folder).
    known: set[str] = set()
    accepts_kwargs = False
    for klass in cls.__mro__[:-1]:  # skip `object`, whose __init__ accepts anything
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        params_ = inspect.signature(init).parameters.values()
        known |= {p.name for p in params_ if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
        accepts_kwargs = any(p.kind is p.VAR_KEYWORD for p in params_)
        if not accepts_kwargs:
            break
    problems = []
    if not accepts_kwargs:
        for key in params:
            if key not in known:
                problems.append(f"{kind} '{name}' does not accept parameter '{key}' (accepts: {sorted(known)})")
    for key in injected:
        if key in params:
            problems.append(f"{kind} '{name}': '{key}' is set by SARIAD and must not be in the config")
    return problems


def validate_config(config: dict) -> list[str]:
    """Check a config without downloading data or building models.

    Resolves every referenced class and verifies that the configured parameters exist on its
    constructor. Returns a list of human readable problems (empty when the config is valid).
    """
    problems: list[str] = []
    for exp in config.get("experiments", []):
        exp_name = exp.get("name", "<unnamed>")
        for key in ("name", "dataset", "model"):
            if key not in exp:
                problems.append(f"experiment '{exp_name}' is missing '{key}'")
        try:
            ds_cls = resolve_dataset(exp["dataset"])
            problems += _check_params("dataset", exp["dataset"], ds_cls, (config.get("datasets") or {}).get(exp["dataset"]) or {})
            model_cls = resolve_model(exp["model"])
            problems += _check_params("model", exp["model"], model_cls, (config.get("models") or {}).get(exp["model"]) or {},
                                      injected=("pre_processor",) if exp.get("preprocessor") else ())
            if exp.get("preprocessor"):
                pre_cls = resolve_preprocessor(exp["preprocessor"])
                problems += _check_params("preprocessor", exp["preprocessor"], pre_cls,
                                          (config.get("preprocessors") or {}).get(exp["preprocessor"]) or {}, injected=("model",))
        except (KeyError, ValueError, ImportError) as e:
            problems.append(f"experiment '{exp_name}': {e}")
    return problems


def run_single_experiment(experiment_config: dict, global_config: dict, run_index: int = 0, runs: int = 1) -> inf.Metrics:
    """Run one training/prediction/metrics pass of an experiment."""
    settings = _settings(global_config)
    exp_name = experiment_config["name"]
    logger.info("Running experiment: %s (run %d/%d)", exp_name, run_index + 1, runs)

    if settings.get("seed") is not None:
        # A different seed per run, otherwise averaging over runs would be pointless.
        seed_everything(int(settings["seed"]) + run_index, workers=True)

    # 1. Datamodule
    dataset_name = experiment_config["dataset"]
    dataset_params = (global_config.get("datasets") or {}).get(dataset_name) or {}
    datamodule = resolve_dataset(dataset_name)(**dataset_params)

    # 2. Model (+ optional preprocessor, which needs the model class to build its transform)
    model_name = experiment_config["model"]
    model_cls = resolve_model(model_name)
    model_params = (global_config.get("models") or {}).get(model_name) or {}

    preprocessor_name = experiment_config.get("preprocessor")
    if preprocessor_name:
        preprocessor_params = (global_config.get("preprocessors") or {}).get(preprocessor_name) or {}
        pre_processor = resolve_preprocessor(preprocessor_name)(model=model_cls, **preprocessor_params)
        model_instance = model_cls(pre_processor=pre_processor, **model_params)
    else:
        model_instance = model_cls(**model_params)

    # 3. Fit & predict (the Engine calls datamodule.setup() itself)
    engine = Engine()
    engine.fit(model=model_instance, datamodule=datamodule)
    torch.cuda.empty_cache()
    predict_results = engine.predict(model=model_instance, datamodule=datamodule)

    # 4. Save predictions and metrics, one folder per run when an experiment is repeated
    output_path = Path(settings.get("output_dir", "results")) / exp_name
    if runs > 1:
        output_path = output_path / f"run_{run_index}"
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "predictions.pkl", "wb") as f:
        pickle.dump(predict_results, f)

    metrics = inf.Metrics(predict_results)
    metrics.save_all(str(output_path))
    logger.info("Metrics saved to %s/", output_path)
    return metrics


def run_experiments(config: dict) -> dict:
    """Run every experiment in ``config`` and write a LaTeX comparison table.

    Can be called directly from another script. Runs that raise are logged and skipped; an
    experiment with no successful run is left out of the comparison table.

    Returns:
        Mapping of experiment name to its list of :class:`SARIAD.utils.inf.Metrics`.
    """
    settings = _settings(config)
    default_runs = int(settings.get("benchmark_runs", 1))
    all_runs_data: dict[str, list] = {}

    for experiment in config.get("experiments", []):
        exp_name = experiment["name"]
        runs = int(experiment.get("runs", default_runs))
        logger.info("Starting experiment group: %s (x%d runs)", exp_name, runs)

        individual_runs = []
        for i in tqdm(range(runs), desc=f"Benchmarking {exp_name}"):
            try:
                individual_runs.append(run_single_experiment(experiment, config, run_index=i, runs=runs))
            except Exception:
                logger.exception("Run %d of %s failed", i + 1, exp_name)
        if individual_runs:
            all_runs_data[exp_name] = individual_runs
        else:
            logger.error("Experiment %s produced no successful runs; excluded from the comparison", exp_name)

    if not all_runs_data:
        raise RuntimeError("No experiment completed successfully; nothing to compare.")

    comparison_data = {name: runs if len(runs) > 1 else runs[0] for name, runs in all_runs_data.items()}
    output_path = Path(settings.get("output_dir", "results"))
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_table = inf.Metrics.compare_multiple_runs(comparison_data, output_dir=str(output_path))
    (output_path / "comparison_table.tex").write_text(comparison_table)
    logger.info("All experiments complete. Comparison table saved to %s/comparison_table.tex", output_path)
    return all_runs_data


def main(argv: list[str] | None = None) -> int:
    """Command line entry point: ``python -m SARIAD.config.run --config <file.yaml>``."""
    parser = argparse.ArgumentParser(description="Run SARIAD experiments from a YAML config file.")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the config (classes and parameters) without downloading data or training.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("The configuration file '%s' was not found.", args.config)
        return 1
    except yaml.YAMLError as e:
        logger.error("Error parsing configuration file: %s", e)
        return 1

    if args.dry_run:
        problems = validate_config(config)
        for problem in problems:
            logger.error(problem)
        if not problems:
            logger.info("Config OK: %d experiment(s) validated.", len(config.get("experiments", [])))
        return 1 if problems else 0

    run_experiments(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
