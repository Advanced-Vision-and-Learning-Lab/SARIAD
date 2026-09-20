"""Train and evaluate one model on one dataset.

    python demo/train.py --dataset MSTAR --model Padim
    python demo/train.py --dataset SSDD --model PadimACE --model-args '{"signature_dir": "datasets/signatures"}'
    python demo/train.py --dataset HRSID --model Padim --preprocessor MedianFilter --output results/metrics

Names are resolved like in the YAML runner (``python -m SARIAD.config.run``): datasets from
``SARIAD.datasets``, models from ``SARIAD.models`` or ``anomalib.models``, preprocessors from
``SARIAD.pre_processing``. Use the YAML runner for repeated runs and comparison tables.
"""

import argparse
import json
import logging

import torch
from anomalib.engine import Engine

from SARIAD.config.run import resolve_dataset, resolve_model, resolve_preprocessor
from SARIAD.utils.inf import Metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="e.g. MSTAR, HRSID, SSDD, SAMPLE_PUBLIC, SARDet_100K")
    parser.add_argument("--model", required=True, help="e.g. Padim, Patchcore, PadimACE, SARATRX, YOLOAnomaly, MSFA")
    parser.add_argument("--preprocessor", default=None, help="e.g. NLM, MedianFilter, SARCNN")
    parser.add_argument("--dataset-args", default="{}", help="JSON dict of datamodule arguments")
    parser.add_argument("--model-args", default="{}", help="JSON dict of model arguments")
    parser.add_argument("--epochs", type=int, default=None, help="Override the model's default number of epochs")
    parser.add_argument("--output", default=None, help="Directory for metrics, plots and a LaTeX table")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision("medium")

    datamodule = resolve_dataset(args.dataset)(**json.loads(args.dataset_args))
    model_cls = resolve_model(args.model)
    model_args = json.loads(args.model_args)
    if args.preprocessor:
        model_args["pre_processor"] = resolve_preprocessor(args.preprocessor)(model=model_cls)
    model = model_cls(**model_args)

    engine = Engine(**({"max_epochs": args.epochs} if args.epochs else {}))
    engine.fit(model=model, datamodule=datamodule)
    torch.cuda.empty_cache()

    # predict the whole dataset with the trained model and compute the SARIAD metrics
    metrics = Metrics(engine.predict(model=model, datamodule=datamodule))
    for name, value in metrics.get_all_metrics().items():
        print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")
    if args.output:
        metrics.save_all(args.output)


if __name__ == "__main__":
    main()
