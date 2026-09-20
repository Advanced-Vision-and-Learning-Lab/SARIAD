"""Evaluation of anomaly detection predictions: streaming metrics, plots and LaTeX tables.

:class:`Inferencer` accumulates the metrics batch by batch with torchmetrics (``update`` /
``compute`` / ``reset``) and can run a model over a datamodule itself. :class:`Metrics` builds on it
to produce the plots and LaTeX tables of the SARIAD benchmarks from saved predictions.
"""

import pickle
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from lightning.pytorch.utilities.types import _PREDICT_OUTPUT
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)

IMAGE_METRICS = [
    "TP", "TN", "FP", "FN", "Accuracy", "Precision", "Recall/Sensitivity", "F1 Score", "Specificity",
    "G-mean", "Missed Alarm Rate (MAR)", "False Alarm Rate (FAR)", "AUROC (Image-level)",
]
PIXEL_METRICS = ["Pixel-level IoU", "Pixel-level F1 Score", "AUROC (Pixel-level)"]


class Inferencer:
    """Streaming image- and pixel-level metrics for anomaly detection, built on torchmetrics.

    Feed it anomalib prediction batches (``Batch`` objects with ``gt_label``, ``pred_label``,
    ``pred_score`` and, for pixel metrics, ``gt_mask``, ``pred_mask`` and ``anomaly_map``) and read the
    metrics with :meth:`compute`::

        inferencer = Inferencer()
        for batch in Engine().predict(model=model, datamodule=datamodule):
            inferencer.update(batch)
        print(inferencer.compute())

    or let it run the model: ``Inferencer().evaluate(model, datamodule)``.

    The pixel AUROC is computed from the continuous ``anomaly_map`` (not the thresholded mask).

    Args:
        pixel_thresholds: ``None`` computes the pixel AUROC exactly, which keeps every pixel score in
            memory. An integer bins the scores into that many thresholds (constant memory, tiny
            approximation error): use it for large datasets.
        device: Device the metrics run on.
    """

    def __init__(self, pixel_thresholds: int | None = None, device: str | torch.device = "cpu") -> None:
        self.label_metrics = MetricCollection({
            "Accuracy": BinaryAccuracy(),
            "Precision": BinaryPrecision(),
            "Recall/Sensitivity": BinaryRecall(),
            "F1 Score": BinaryF1Score(),
            "Specificity": BinarySpecificity(),
            "confusion_matrix": BinaryConfusionMatrix(),
        })
        self.image_auroc = BinaryAUROC()
        self.mask_metrics = MetricCollection({"Pixel-level IoU": BinaryJaccardIndex(), "Pixel-level F1 Score": BinaryF1Score()})
        self.pixel_auroc = BinaryAUROC(thresholds=pixel_thresholds)
        self.to(device)
        self._has_pixels = False
        self._images = [0, 0]  # normal / anomalous images seen
        self._pixels = [0, 0]  # normal / anomalous pixels seen

    def to(self, device: str | torch.device) -> "Inferencer":
        self.device = torch.device(device)
        for metric in (self.label_metrics, self.image_auroc, self.mask_metrics, self.pixel_auroc):
            metric.to(self.device)
        return self

    def reset(self) -> None:
        for metric in (self.label_metrics, self.image_auroc, self.mask_metrics, self.pixel_auroc):
            metric.reset()
        self._has_pixels = False
        self._images, self._pixels = [0, 0], [0, 0]

    @torch.no_grad()
    def update(self, batch) -> None:
        """Accumulate one batch of predictions (an anomalib ``Batch``)."""
        gt_label = batch.gt_label.to(self.device).long()
        self._images[0] += int((gt_label == 0).sum())
        self._images[1] += int((gt_label == 1).sum())
        self.label_metrics.update(batch.pred_label.to(self.device).long(), gt_label)
        self.image_auroc.update(batch.pred_score.to(self.device).float().reshape(-1), gt_label)

        if batch.gt_mask is not None and batch.pred_mask is not None:
            gt_mask = batch.gt_mask.to(self.device).long()
            self._pixels[0] += int((gt_mask == 0).sum())
            self._pixels[1] += int((gt_mask == 1).sum())
            self.mask_metrics.update(batch.pred_mask.to(self.device).long(), gt_mask)
            if batch.anomaly_map is not None:
                self.pixel_auroc.update(batch.anomaly_map.to(self.device).float().reshape(gt_mask.shape), gt_mask)
            self._has_pixels = True

    def compute(self) -> dict:
        """All metrics accumulated so far; metrics that cannot be computed are reported as ``"Error: ..."``."""
        results: dict = {}

        def attempt(name: str, fn) -> None:
            try:
                results[name] = fn()
            except Exception as e:  # e.g. only one class present
                results[name] = f"Error: {e}"

        scores = self.label_metrics.compute()
        tn, fp, fn, tp = (int(v) for v in scores["confusion_matrix"].flatten().tolist())
        results.update({"TP": tp, "TN": tn, "FP": fp, "FN": fn})
        for name in ("Accuracy", "Precision", "Recall/Sensitivity", "F1 Score", "Specificity"):
            results[name] = scores[name].item()
        recall, specificity = results["Recall/Sensitivity"], results["Specificity"]
        results["G-mean"] = sqrt(recall * specificity) if recall > 0 and specificity > 0 else 0
        results["Missed Alarm Rate (MAR)"] = 1 - recall
        results["False Alarm Rate (FAR)"] = 1 - specificity
        needs_both = "N/A (needs normal and anomalous samples)"
        if min(self._images) == 0:
            results["AUROC (Image-level)"] = needs_both
        else:
            attempt("AUROC (Image-level)", lambda: self.image_auroc.compute().item())

        if self._has_pixels:
            pixel = self.mask_metrics.compute()
            results["Pixel-level IoU"], results["Pixel-level F1 Score"] = pixel["Pixel-level IoU"].item(), pixel["Pixel-level F1 Score"].item()
            if min(self._pixels) == 0:
                results["AUROC (Pixel-level)"] = needs_both
            else:
                attempt("AUROC (Pixel-level)", lambda: self.pixel_auroc.compute().item())
        else:
            results.update({name: "N/A (no masks)" for name in PIXEL_METRICS})
        return results

    def evaluate(self, model, datamodule, ckpt_path: str | None = None, engine=None) -> dict:
        """Run ``model`` over the datamodule's prediction data and return the metrics.

        Args:
            model: Trained anomalib model.
            datamodule: Datamodule providing the prediction/test data.
            ckpt_path: Optional checkpoint to load before predicting.
            engine: An ``anomalib.engine.Engine`` to use (a default one is created otherwise).
        """
        from anomalib.engine import Engine

        self.reset()
        for batch in (engine or Engine()).predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path):
            self.update(batch)
        return self.compute()

    @classmethod
    def from_predictions(cls, predictions: _PREDICT_OUTPUT, **kwargs) -> "Inferencer":
        inferencer = cls(**kwargs)
        for batch in predictions:
            inferencer.update(batch)
        return inferencer


class Metrics:
    """
    A class to compute and visualize metrics for SARIAD model predictions.

    Args:
        predictions (_PREDICT_OUTPUT, optional): The raw prediction output from a PyTorch Lightning Trainer.
                                                 Defaults to None.
        metrics_to_calculate (list[str], optional): A list of metric names to calculate. Defaults to all metrics.
        pixel_thresholds (int, optional): Bin the pixel AUROC into this many thresholds (constant memory); ``None`` is exact.
    """
    def __init__(self, predictions: _PREDICT_OUTPUT = None, metrics_to_calculate: list[str] = None, pixel_thresholds: int | None = None):
        self.predictions = predictions
        self.pixel_thresholds = pixel_thresholds
        self.inferencer = None
        self._all_metrics = None
        self.gt_labels = None
        self.pred_labels = None
        self.pred_scores = None

        self.available_metrics = IMAGE_METRICS + PIXEL_METRICS
        
        if metrics_to_calculate is None:
            self.metrics_to_calculate = list(self.available_metrics)
        else:
            invalid_metrics = [m for m in metrics_to_calculate if m not in self.available_metrics]
            if invalid_metrics:
                raise ValueError(f"Invalid metrics requested: {invalid_metrics}. Available metrics are: {self.available_metrics}")
            self.metrics_to_calculate = metrics_to_calculate

        if self.predictions is not None:
            self._aggregate_predictions()

    def _aggregate_predictions(self):
        """Stream the predictions through an :class:`Inferencer` and keep the image-level tensors (for the curves)."""
        self.inferencer = Inferencer(pixel_thresholds=self.pixel_thresholds)
        gt_labels, pred_labels, pred_scores = [], [], []
        for batch in self.predictions:
            self.inferencer.update(batch)
            gt_labels.append(batch.gt_label)
            pred_labels.append(batch.pred_label)
            pred_scores.append(batch.pred_score)

        self.gt_labels = torch.cat(gt_labels)
        self.pred_labels = torch.cat(pred_labels)
        self.pred_scores = torch.cat(pred_scores)

    @classmethod
    def from_pickle(cls, prediction_path: Path, metrics_to_calculate: list[str] = None, pixel_thresholds: int | None = None):
        """
        Creates a Metrics instance by reading predictions from a pickle file.

        Args:
            prediction_path (Path): The path to the pickle file containing the predictions.
            metrics_to_calculate (list[str], optional): A list of metric names to calculate.

        Returns:
            Metrics: An instance of the class with loaded predictions.
        """
        with open(prediction_path, "rb") as f:
            predictions = pickle.load(f)
        return cls(predictions, metrics_to_calculate, pixel_thresholds)

    def get_all_metrics(self) -> dict:
        """
        Calculates and returns a dictionary of all classification and segmentation metrics.

        Returns:
            dict: A dictionary containing all computed metrics.
        """
        if self.predictions is None:
            raise ValueError("Predictions are not loaded. Use from_pickle() or provide predictions to the constructor.")

        if self._all_metrics is None:
            self._all_metrics = self.inferencer.compute()
        return {name: self._all_metrics[name] for name in self.metrics_to_calculate}

    def save_all(self, output_dir: str = "."):
        """
        Generates and saves all metric plots, a text file of the metrics dictionary,
        and a LaTeX table of the metrics.
        
        Args:
            output_dir (str): The directory to save the output files in.
        """
        if self.predictions is None:
            raise ValueError("Predictions are not loaded. Cannot plot.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        metrics = self.get_all_metrics()

        # Save metrics to a text file
        with open(output_path / 'metrics.txt', 'w') as f:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        # Save LaTeX table to a .tex file
        latex_table = self._to_latex_table_single_run()
        with open(output_path / 'metrics_table.tex', 'w') as f:
            f.write(latex_table)

        # Plot 1: Confusion Matrix
        confmat = BinaryConfusionMatrix()(self.pred_labels, self.gt_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(confmat.numpy(), annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(output_path / 'confusion_matrix.png')
        plt.close()

        # Plot 2: ROC Curve
        fpr, tpr, _ = roc_curve(self.gt_labels.cpu().numpy(), self.pred_scores.cpu().numpy())
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(output_path / 'roc_curve.png')
        plt.close()

        # Plot 3: Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.gt_labels.cpu().numpy(), self.pred_scores.cpu().numpy())
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="upper right")
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.savefig(output_path / 'pr_curve.png')
        plt.close()

    def _to_latex_table_single_run(self) -> str:
        """
        Generates a LaTeX table string for all computed metrics of a single run.
        This is a private helper method.

        Returns:
            str: The LaTeX table code as a string.
        """
        metrics = self.get_all_metrics()
        
        latex_str = "\\begin{table}[h!]\n"
        latex_str += "\\centering\n"
        latex_str += "\\begin{tabular}{|l|l|}\n"
        latex_str += "\\hline\n"
        latex_str += "Metric & Value \\\\\n"
        latex_str += "\\hline\n"
        
        for metric_name, value in metrics.items():
            display_name = self._get_table_rows().get(metric_name, metric_name)
            if isinstance(value, float):
                latex_str += f"{display_name} & ${value:.4f}$ \\\\\n"
            else:
                latex_str += f"{display_name} & {value} \\\\\n"
        
        latex_str += "\\hline\n"
        latex_str += "\\end{tabular}\n"
        latex_str += "\\caption{Metrics for a Single Run}\n"
        latex_str += "\\label{tab:metrics_single_run}\n"
        latex_str += "\\end{table}\n"
        
        return latex_str

    @staticmethod
    def _get_table_rows() -> dict:
        """Defines the order and formatting of metrics for the LaTeX table."""
        return {
            "TP": "True Positives", "TN": "True Negatives", "FP": "False Positives", "FN": "False Negatives",
            "Accuracy": "Accuracy", "Precision": "Precision", "Recall/Sensitivity": "Recall / Sensitivity",
            "F1 Score": "F1 Score", "Specificity": "Specificity", "G-mean": "G-mean",
            "Missed Alarm Rate (MAR)": "Missed Alarm Rate (MAR)", "False Alarm Rate (FAR)": "False Alarm Rate (FAR)",
            "AUROC (Image-level)": "AUROC (Image-level)",
            "Pixel-level IoU": "Pixel-level IoU", "Pixel-level F1 Score": "Pixel-level F1 Score",
            "AUROC (Pixel-level)": "AUROC (Pixel-level)",
        }
    
    @staticmethod
    def _is_better(metric_name: str, val1, val2) -> bool:
        """Determines if val1 is a better metric value than val2."""
        lower_is_better = ["FP", "FN", "Missed Alarm Rate (MAR)", "False Alarm Rate (FAR)"]
        if metric_name in lower_is_better:
            return val1 < val2
        else:
            return val1 > val2
        
    def _plot_roc_curve(self, run_data: dict[str, "Metrics | list"], output_dir: str = "."):
        """
        Generates and saves a single ROC curve plot, showing the average curve and variability
        if multiple runs are provided for a single model.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        
        for name, data in run_data.items():
            if isinstance(data, list):
                # Calculate mean ROC and standard deviation for a list of runs
                all_fprs = []
                all_tprs = []
                all_aucs = []
                
                # We need to use a common set of false positive rates to average the true positive rates
                mean_fpr = np.linspace(0, 1, 100)

                for run in data:
                    fpr, tpr, _ = roc_curve(run.gt_labels.cpu().numpy(), run.pred_scores.cpu().numpy())
                    all_fprs.append(fpr)
                    all_tprs.append(tpr)
                    all_aucs.append(auc(fpr, tpr))
                    
                interp_tprs = []
                for idx in range(len(data)):
                    interp_tpr = np.interp(mean_fpr, all_fprs[idx], all_tprs[idx])
                    interp_tpr[0] = 0.0
                    interp_tprs.append(interp_tpr)
                
                mean_tpr = np.mean(interp_tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)
                std_auc = np.std(all_aucs)
                std_tpr = np.std(interp_tprs, axis=0)
                
                tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                
                # Plot mean ROC curve
                plt.plot(mean_fpr, mean_tpr, lw=2, label=rf'{name} (Mean AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})')
                
                # Plot variability area
                plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=0.2, label=r"$\pm$ 1 std. dev.")
            
            else:
                # Plot single ROC curve
                fpr, tpr, _ = roc_curve(data.gt_labels.cpu().numpy(), data.pred_scores.cpu().numpy())
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance Level')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.savefig(output_path / 'roc_curve_comparison.png')
        plt.close()

    @classmethod
    def compare_multiple_runs(cls, runs_data: dict[str, "Metrics | list"], output_dir: str = "comparison_output") -> str:
        """
        Generates a single LaTeX table comparing multiple runs, with the best value bolded,
        and plots a combined ROC curve.
        
        Args:
            runs_data (dict[str, "Metrics" | list]): A dictionary where keys are run names and values are either
                                                      a single Metrics object or a list of Metrics objects.
            output_dir (str): Where the combined ROC plot is saved.

        Returns:
            str: The LaTeX table code as a string.
        """
        if not runs_data:
            return ""

        # Plot the combined ROC curve
        metrics_instance = cls()
        metrics_instance._plot_roc_curve(runs_data, output_dir=output_dir)

        run_names = list(runs_data.keys())
        processed_metrics_data = {}
        for name, data in runs_data.items():
            if isinstance(data, list):
                # Calculate average and std dev for a list of runs
                all_run_metrics = [run.get_all_metrics() for run in data]
                avg_metrics = {}
                std_dev_metrics = {}
                # Use a stable key order from the first run's metrics
                first_metrics_keys = next(iter(all_run_metrics)).keys() if all_run_metrics else []
                for metric_key in first_metrics_keys:
                    values = [m.get(metric_key) for m in all_run_metrics if isinstance(m.get(metric_key), (int, float))]
                    if values:
                        avg_metrics[metric_key] = np.mean(values)
                        std_dev_metrics[metric_key] = np.std(values)
                processed_metrics_data[name] = (avg_metrics, std_dev_metrics)
            else:
                # Get metrics for a single run
                processed_metrics_data[name] = (data.get_all_metrics(), None)

        first_run_metrics = next(iter(processed_metrics_data.values()))[0]
        metrics_to_compare = first_run_metrics.keys()
        
        table_rows = cls._get_table_rows()
        
        # Determine best average values for each metric across all runs
        best_values = {}
        for metric_key in metrics_to_compare:
            metric_values = [processed_metrics_data[name][0].get(metric_key) for name in run_names]
            
            best_val = None
            for val in metric_values:
                if isinstance(val, (int, float)):
                    if best_val is None or cls._is_better(metric_key, val, best_val):
                        best_val = val
            best_values[metric_key] = best_val

        # Build the LaTeX string
        latex_str = "\\begin{table}[h!]\n"
        latex_str += "\\centering\n"
        latex_str += "\\begin{tabular}{|l|" + "|l" * len(run_names) + "|}\n"
        latex_str += "\\hline\n"
        
        # Header row
        header = ["Metric"] + run_names
        latex_str += " & ".join(header) + " \\\\\n"
        latex_str += "\\hline\n"
        
        # Metric rows
        for metric_key in metrics_to_compare:
            display_name = table_rows.get(metric_key, metric_key)
            row_values = []
            for run_name in run_names:
                avg_metrics, std_dev_metrics = processed_metrics_data[run_name]
                val = avg_metrics.get(metric_key)
                
                if val is None:
                    formatted_val = "N/A"
                else:
                    # Format value, include std dev if available, and bold if it's the best
                    formatted_val = f"${val:.4f}$"
                    if std_dev_metrics and std_dev_metrics.get(metric_key) is not None:
                        formatted_val = rf"${val:.4f} \pm {std_dev_metrics.get(metric_key):.4f}$"

                    if val == best_values.get(metric_key):
                        formatted_val = f"\\textbf{{{formatted_val}}}"
                row_values.append(formatted_val)
            
            latex_str += f"{display_name} & " + " & ".join(row_values) + " \\\\\n"
        
        latex_str += "\\hline\n"
        latex_str += "\\end{tabular}\n"
        latex_str += "\\caption{Comparison of Multiple Model Runs}\n"
        latex_str += "\\label{tab:model_comparison}\n"
        latex_str += "\\end{table}\n"

        return latex_str
