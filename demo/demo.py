from SARIAD.config.run import run_experiments
import yaml
from pathlib import Path

if __name__ == "__main__":
    config_file_path = "demo.yaml" 
    try:
        with open(config_file_path, "r") as f:
            file_config = yaml.safe_load(f)
        run_experiments(file_config)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_file_path}")
