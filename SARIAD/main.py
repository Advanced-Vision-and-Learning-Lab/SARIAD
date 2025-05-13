import argparse
import yaml
import pytorch_lightning as pl
from .datasets import *
from models import autoencoder, transformer, anomalib_model
from benchmarks.benchmarking import benchmark

# Load configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Select model
def get_model(model_name, input_size):
    if model_name in ['autoencoder','transformer']:
        if model_name == 'autoencoder':
            return autoencoder.Autoencoder()
        elif model_name == 'transformer':
            return transformer.Transformer()
    else:
        return anomalib_model.AnomalibModel(model_name=model_name, input_size=input_size)

# Main function
def main():
    parser = argparse.ArgumentParser(description='SARIAD - SAR Anomaly Detection')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    args = parser.parse_args()

    config = load_config(args.config)

    # Set random seed
    pl.seed_everything(config['seed'], workers=True)

    # Initialize data module
    data_module = SARDataModule(
        data_dir=config['data']['path'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )

    # Initialize model
    model = get_model(config['model']['name'], tuple(config['model']['input_size']))

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        log_every_n_steps=config['logging']['log_interval'],
        default_root_dir=config['logging']['log_dir']
    )

    # Train model
    trainer.fit(model, data_module)

    # Benchmark if enabled
    if config['benchmark']['enabled']:
        benchmark(model, data_module, num_runs=config['benchmark']['num_runs'])

if __name__ == '__main__':
    main()
