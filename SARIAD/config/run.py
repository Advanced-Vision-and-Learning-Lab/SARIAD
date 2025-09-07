import yaml
import importlib
import torch
import pickle
from pathlib import Path
from tqdm import tqdm
from anomalib.engine import Engine
from SARIAD.utils import inf
import argparse
from typing import Union

def get_full_class_path(class_name: str, component_type: str) -> str:
    """Returns the full module path for a given component."""
    if component_type == "datasets":
        return f"SARIAD.datasets.datamodules.image.{class_name.lower()}.{class_name.lower()}"
    
    if component_type == "models":
        if class_name in ["SARATRX"]:
            return f"SARIAD.models.{class_name}"
        else:
            return f"anomalib.models.image.{class_name.lower()}"
    
    if component_type == "preprocessors":
        # The corrected path to the preprocessor module file
        return f"SARIAD.pre_processing.{class_name}.{class_name}"
    
    raise ValueError(f"Unknown component type: {component_type}")

def import_class(module_path: str, class_name: str):
    """Dynamically imports a class from a module."""
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def run_single_experiment(experiment_config: dict, global_config: dict) -> inf.Metrics:
    """
    Runs a single experiment based on the provided configuration.
    """
    exp_name = experiment_config["name"]
    print(f"\nRunning experiment: {exp_name}")

    # 1. Load Datamodule
    dataset_name = experiment_config["dataset"]
    dataset_cls = import_class(get_full_class_path(dataset_name, "datasets"), dataset_name)
    dataset_params = global_config["datasets"].get(dataset_name, {})
    datamodule = dataset_cls(**dataset_params)
    datamodule.setup()

    # 2. Load Model and Preprocessor
    model_name = experiment_config["model"]
    model_cls = import_class(get_full_class_path(model_name, "models"), model_name)
    model_params = global_config["models"].get(model_name, {})
    model_params = model_params if model_params is not None else {}
    
    preprocessor_name = experiment_config["preprocessor"]
    if preprocessor_name:
        preprocessor_cls = import_class(get_full_class_path(preprocessor_name, "preprocessors"), preprocessor_name)
        
        # Get preprocessor parameters from the config
        preprocessor_params = global_config["preprocessors"].get(preprocessor_name)
        preprocessor_params = preprocessor_params if preprocessor_params is not None else {}
        
        # Preprocessor requires the model class for instantiation
        pre_processor = preprocessor_cls(model=model_cls, **preprocessor_params)
        model_instance = model_cls(pre_processor=pre_processor, **model_params)
    else:
        model_instance = model_cls(**model_params)

    # 3. Run Engine (Fit & Predict)
    engine = Engine()
    engine.fit(model=model_instance, datamodule=datamodule)
    torch.cuda.empty_cache()

    predict_results = engine.predict(
        model=model_instance,
        datamodule=datamodule,
    )
    
    # Save predictions for metric calculation and potential debugging
    output_path = Path(global_config["output_dir"]) / exp_name
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "predictions.pkl", "wb") as f:
        pickle.dump(predict_results, f)

    # 4. Compute and Save Metrics
    metrics = inf.Metrics(predict_results)
    metrics.save_all(str(output_path))
    print(f"Metrics saved to {output_path}/")
    return metrics

def run_experiments(config: dict):
    """
    Orchestrates the experiments based on a configuration dictionary.
    This function can be called directly from another script.
    """
    global_config = config.get("global", {})
    experiments_list = config.get("experiments", [])
    benchmark_runs = config.get("benchmark_runs", 1)
    
    all_runs_data = {}

    for experiment in experiments_list:
        exp_name = experiment["name"]
        print(f"\n--- Starting experiment group: {exp_name} (x{benchmark_runs} runs) ---")
        
        individual_runs = []
        for i in tqdm(range(benchmark_runs), desc=f"Benchmarking {exp_name}"):
            try:
                metrics = run_single_experiment(experiment, config)
                individual_runs.append(metrics)
            except Exception as e:
                print(f"An error occurred during run {i+1} of {exp_name}: {e}")
                continue
        
        all_runs_data[exp_name] = individual_runs
        
    print("\n--- Generating Comparison Table ---")
    comparison_data = {
        name: metrics_list if len(metrics_list) > 1 else metrics_list[0]
        for name, metrics_list in all_runs_data.items()
    }

    comparison_table = inf.Metrics.compare_multiple_runs(comparison_data)
    
    output_path = Path(global_config.get("output_dir", "."))
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "comparison_table.tex", "w") as f:
        f.write(comparison_table)
        
    print(f"All experiments complete. Comparison table saved to {output_path}/comparison_table.tex")

def main():
    """
    Parses command-line arguments and loads the config, then calls run_experiments.
    """
    parser = argparse.ArgumentParser(description="Run SARIAD experiments from a YAML config file.")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()
    
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: The configuration file '{args.config}' was not found.")
        return
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return

    run_experiments(config)
    
if __name__ == "__main__":
    main()
