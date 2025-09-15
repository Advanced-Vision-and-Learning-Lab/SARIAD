import torch
import pickle
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryAUROC,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
    BinaryConfusionMatrix,
)
from pathlib import Path
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from lightning.pytorch.utilities.types import _PREDICT_OUTPUT
import numpy as np

class Metrics:
    """
    A class to compute and visualize metrics for SARIAD model predictions.

    Args:
        predictions (_PREDICT_OUTPUT, optional): The raw prediction output from a PyTorch Lightning Trainer.
                                                 Defaults to None.
        metrics_to_calculate (list[str], optional): A list of metric names to calculate. Defaults to all metrics.
    """
    def __init__(self, predictions: _PREDICT_OUTPUT = None, metrics_to_calculate: list[str] = None):
        self.predictions = predictions
        self.gt_labels = None
        self.pred_labels = None
        self.gt_masks = None
        self.pred_masks = None
        self.pred_scores = None

        self.available_metrics = {
            "TP": self._calc_tp, "TN": self._calc_tn, "FP": self._calc_fp, "FN": self._calc_fn,
            "Accuracy": self._calc_accuracy, "Precision": self._calc_precision,
            "Recall/Sensitivity": self._calc_recall, "F1 Score": self._calc_f1_score,
            "Specificity": self._calc_specificity, "G-mean": self._calc_g_mean,
            "Missed Alarm Rate (MAR)": self._calc_mar, "False Alarm Rate (FAR)": self._calc_far,
            "AUROC (Image-level)": self._calc_auroc,
            "Pixel-level IoU": self._calc_iou, "Pixel-level F1 Score": self._calc_f1_seg,
            "AUROC (Pixel-level)": self._calc_auroc_seg,
        }
        
        if metrics_to_calculate is None:
            self.metrics_to_calculate = list(self.available_metrics.keys())
        else:
            invalid_metrics = [m for m in metrics_to_calculate if m not in self.available_metrics]
            if invalid_metrics:
                raise ValueError(f"Invalid metrics requested: {invalid_metrics}. Available metrics are: {list(self.available_metrics.keys())}")
            self.metrics_to_calculate = metrics_to_calculate

        if self.predictions is not None:
            self._aggregate_predictions()

    def _aggregate_predictions(self):
        """Aggregates and concatenates prediction results from all batches."""
        all_gt_labels = []
        all_pred_labels = []
        all_gt_masks = []
        all_pred_masks = []
        all_pred_scores = []

        for batch in self.predictions:
            all_gt_labels.append(batch.gt_label)
            all_pred_labels.append(batch.pred_label)
            all_gt_masks.append(batch.gt_mask.to(torch.float32))
            all_pred_masks.append(batch.pred_mask.to(torch.float32))
            all_pred_scores.append(batch.pred_score)

        self.gt_labels = torch.cat(all_gt_labels)
        self.pred_labels = torch.cat(all_pred_labels)
        self.gt_masks = torch.cat(all_gt_masks)
        self.pred_masks = torch.cat(all_pred_masks)
        self.pred_scores = torch.cat(all_pred_scores)

    @classmethod
    def from_pickle(cls, prediction_path: Path, metrics_to_calculate: list[str] = None):
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
        return cls(predictions, metrics_to_calculate)

    # Helper methods for calculating each metric
    def _calc_conf_matrix(self):
        if not hasattr(self, '_confmat'):
            self._confmat = BinaryConfusionMatrix()(self.pred_labels, self.gt_labels)
        return self._confmat.flatten().tolist()
    
    def _calc_tp(self): return self._calc_conf_matrix()[3]
    def _calc_tn(self): return self._calc_conf_matrix()[0]
    def _calc_fp(self): return self._calc_conf_matrix()[1]
    def _calc_fn(self): return self._calc_conf_matrix()[2]

    def _calc_accuracy(self): return BinaryAccuracy()(self.pred_labels, self.gt_labels).item()
    def _calc_precision(self): return BinaryPrecision()(self.pred_labels, self.gt_labels).item()
    def _calc_recall(self): return BinaryRecall()(self.pred_labels, self.gt_labels).item()
    def _calc_f1_score(self): return BinaryF1Score()(self.pred_labels, self.gt_labels).item()
    def _calc_specificity(self): return BinarySpecificity()(self.pred_labels, self.gt_labels).item()
    def _calc_g_mean(self):
        recall = self._calc_recall()
        specificity = self._calc_specificity()
        return sqrt(recall * specificity) if recall > 0 and specificity > 0 else 0
    def _calc_mar(self): return 1 - self._calc_recall()
    def _calc_far(self): return 1 - self._calc_specificity()
    def _calc_auroc(self): return BinaryAUROC()(self.pred_scores, self.gt_labels).item()
    def _calc_iou(self):
        gt_masks_flat = self.gt_masks.view(-1)
        pred_masks_flat = self.pred_masks.view(-1)
        return BinaryJaccardIndex()(pred_masks_flat, gt_masks_flat.long()).item()
    def _calc_f1_seg(self):
        gt_masks_flat = self.gt_masks.view(-1)
        pred_masks_flat = self.pred_masks.view(-1)
        return BinaryF1Score()(pred_masks_flat, gt_masks_flat.long()).item()
    def _calc_auroc_seg(self):
        gt_masks_flat = self.gt_masks.view(-1)
        pred_masks_flat = self.pred_masks.view(-1)
        return BinaryAUROC()(pred_masks_flat, gt_masks_flat.long()).item()

    def get_all_metrics(self) -> dict:
        """
        Calculates and returns a dictionary of all classification and segmentation metrics.

        Returns:
            dict: A dictionary containing all computed metrics.
        """
        if self.predictions is None:
            raise ValueError("Predictions are not loaded. Use from_pickle() or provide predictions to the constructor.")

        metrics = {}
        for metric_name in self.metrics_to_calculate:
            if metric_name in self.available_metrics:
                try:
                    metrics[metric_name] = self.available_metrics[metric_name]()
                except Exception as e:
                    metrics[metric_name] = f"Error: {e}"
        
        return metrics

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
        output_path.mkdir(exist_ok=True)
        
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
        output_path.mkdir(exist_ok=True)
        
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
                plt.plot(mean_fpr, mean_tpr, lw=2, label=f'{name} (Mean AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})')
                
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
    def compare_multiple_runs(cls, runs_data: dict[str, "Metrics | list"]) -> str:
        """
        Generates a single LaTeX table comparing multiple runs, with the best value bolded,
        and plots a combined ROC curve.
        
        Args:
            runs_data (dict[str, "Metrics" | list]): A dictionary where keys are run names and values are either
                                                      a single Metrics object or a list of Metrics objects.

        Returns:
            str: The LaTeX table code as a string.
        """
        if not runs_data:
            return ""

        # Plot the combined ROC curve
        metrics_instance = cls()
        metrics_instance._plot_roc_curve(runs_data, output_dir="comparison_output")

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
                        formatted_val = f"${val:.4f} \pm {std_dev_metrics.get(metric_key):.4f}$"

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

def example():
    try:
        # Example 1: A single run
        run1 = Metrics.from_pickle(Path("results/run1_predictions.pkl"))
        run1.save_all(output_dir="run1_output")
        print("Metrics and plots for Run 1 saved to run1_output/")

        # Example 2: A list of runs for one model to show average and std dev
        run_list_A = [
            Metrics.from_pickle(Path("results/run_A_1.pkl")),
            Metrics.from_pickle(Path("results/run_A_2.pkl")),
            Metrics.from_pickle(Path("results/run_A_3.pkl")),
        ]
        
        # Example 3: A single run for another model
        run_B = Metrics.from_pickle(Path("results/run_B_1.pkl"))

        runs_to_compare = {
            "Model A": run_list_A,
            "Model B": run_B,
        }
        
        comparison_table = Metrics.compare_multiple_runs(runs_to_compare)
        print("\n--- Comparison LaTeX Table ---\n")
        print(comparison_table)
        
        with open("comparison_table.tex", "w") as f:
            f.write(comparison_table)
        print("\nComparison table saved to comparison_table.tex")
        
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the prediction pickle files exist.")
