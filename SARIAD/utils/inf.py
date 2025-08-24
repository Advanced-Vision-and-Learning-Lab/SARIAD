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

class Metrics:
    """
    A class to compute and visualize metrics for SARIAD model predictions.

    Args:
        predictions (_PREDICT_OUTPUT, optional): The raw prediction output from a PyTorch Lightning Trainer.
                                               Defaults to None.
    """
    def __init__(self, predictions: _PREDICT_OUTPUT = None):
        self.predictions = predictions
        self.gt_labels = None
        self.pred_labels = None
        self.gt_masks = None
        self.pred_masks = None
        self.pred_scores = None

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
    def from_pickle(cls, prediction_path: Path):
        """
        Creates a Metrics instance by reading predictions from a pickle file.

        Args:
            prediction_path (Path): The path to the pickle file containing the predictions.

        Returns:
            Metrics: An instance of the class with loaded predictions.
        """
        with open(prediction_path, "rb") as f:
            predictions = pickle.load(f)
        return cls(predictions)

    def get_all_metrics(self) -> dict:
        """
        Calculates and returns a dictionary of all classification and segmentation metrics.

        Returns:
            dict: A dictionary containing all computed metrics.
        """
        if self.predictions is None:
            raise ValueError("Predictions are not loaded. Use from_pickle() or provide predictions to the constructor.")

        # Calculate confusion matrix components
        confmat = BinaryConfusionMatrix()(self.pred_labels, self.gt_labels)
        tn, fp, fn, tp = confmat.flatten().tolist()
        
        # Calculate classification metrics
        accuracy = BinaryAccuracy()(self.pred_labels, self.gt_labels).item()
        f1_score = BinaryF1Score()(self.pred_labels, self.gt_labels).item()
        auroc = BinaryAUROC()(self.pred_scores, self.gt_labels).item()
        precision = BinaryPrecision()(self.pred_labels, self.gt_labels).item()
        recall = BinaryRecall()(self.pred_labels, self.gt_labels).item()
        specificity = BinarySpecificity()(self.pred_labels, self.gt_labels).item()
        
        # Calculate derived metrics
        g_mean = sqrt(recall * specificity) if recall > 0 and specificity > 0 else 0
        mar = 1 - recall
        far = 1 - specificity

        # Calculate segmentation metrics
        gt_masks_flat = self.gt_masks.view(-1)
        pred_masks_flat = self.pred_masks.view(-1)
        iou = BinaryJaccardIndex()(pred_masks_flat, gt_masks_flat.long()).item()
        f1_seg = BinaryF1Score()(pred_masks_flat, gt_masks_flat.long()).item()
        auroc_seg = BinaryAUROC()(pred_masks_flat, gt_masks_flat.long()).item()
        
        metrics = {
            "classification": {
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall/Sensitivity": recall,
                "F1 Score": f1_score,
                "Specificity": specificity,
                "G-mean": g_mean,
                "Missed Alarm Rate (MAR)": mar,
                "False Alarm Rate (FAR)": far,
                "AUROC (Image-level)": auroc,
            },
            "segmentation": {
                "Pixel-level IoU": iou,
                "Pixel-level F1 Score": f1_seg,
                "AUROC (Pixel-level)": auroc_seg,
            }
        }
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
            f.write("--- Classification Metrics ---\n")
            for key, value in metrics["classification"].items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            f.write("\n--- Segmentation Metrics ---\n")
            for key, value in metrics["segmentation"].items():
                f.write(f"{key}: {value:.4f}\n")
        
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
        
        # Classification Metrics
        classification_metrics = metrics["classification"]
        classification_header = ["Metric", "Value"]
        classification_rows = [
            ["True Positives", f"${classification_metrics['TP']}$"],
            ["True Negatives", f"${classification_metrics['TN']}$"],
            ["False Positives", f"${classification_metrics['FP']}$"],
            ["False Negatives", f"${classification_metrics['FN']}$"],
            ["Accuracy", f"${classification_metrics['Accuracy']:.4f}$"],
            ["Precision", f"${classification_metrics['Precision']:.4f}$"],
            ["Recall / Sensitivity", f"${classification_metrics['Recall/Sensitivity']:.4f}$"],
            ["F1 Score", f"${classification_metrics['F1 Score']:.4f}$"],
            ["Specificity", f"${classification_metrics['Specificity']:.4f}$"],
            ["G-mean", f"${classification_metrics['G-mean']:.4f}$"],
            ["Missed Alarm Rate (MAR)", f"${classification_metrics['Missed Alarm Rate (MAR)']:.4f}$"],
            ["False Alarm Rate (FAR)", f"${classification_metrics['False Alarm Rate (FAR)']:.4f}$"],
            ["AUROC (Image-level)", f"${classification_metrics['AUROC (Image-level)']:.4f}$"],
        ]

        # Segmentation Metrics
        segmentation_metrics = metrics["segmentation"]
        segmentation_header = ["Metric", "Value"]
        segmentation_rows = [
            ["Pixel-level IoU", f"${segmentation_metrics['Pixel-level IoU']:.4f}$"],
            ["Pixel-level F1 Score", f"${segmentation_metrics['Pixel-level F1 Score']:.4f}$"],
            ["AUROC (Pixel-level)", f"${segmentation_metrics['AUROC (Pixel-level)']:.4f}$"],
        ]

        # Build the LaTeX string
        latex_str = ""
        
        # Classification Table
        latex_str += "\\begin{table}[h!]\n"
        latex_str += "\\centering\n"
        latex_str += "\\begin{tabular}{|l|l|}\n"
        latex_str += "\\hline\n"
        latex_str += f"{' & '.join(classification_header)} \\\\\n"
        latex_str += "\\hline\n"
        for row in classification_rows:
            latex_str += f"{' & '.join(row)} \\\\\n"
        latex_str += "\\hline\n"
        latex_str += "\\end{tabular}\n"
        latex_str += "\\caption{Classification Metrics}\n"
        latex_str += "\\label{tab:classification_metrics}\n"
        latex_str += "\\end{table}\n"
        latex_str += "\n"

        # Segmentation Table
        latex_str += "\\begin{table}[h!]\n"
        latex_str += "\\centering\n"
        latex_str += "\\begin{tabular}{|l|l|}\n"
        latex_str += "\\hline\n"
        latex_str += f"{' & '.join(segmentation_header)} \\\\\n"
        latex_str += "\\hline\n"
        for row in segmentation_rows:
            latex_str += f"{' & '.join(row)} \\\\\n"
        latex_str += "\\hline\n"
        latex_str += "\\end{tabular}\n"
        latex_str += "\\caption{Segmentation Metrics}\n"
        latex_str += "\\label{tab:segmentation_metrics}\n"
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
        
    @classmethod
    def compare_multiple_runs(cls, runs: dict[str, "Metrics"]) -> str:
        """
        Generates a single LaTeX table comparing multiple runs, with the best value bolded.
        
        Args:
            runs (dict[str, Metrics]): A dictionary where keys are run names and values are Metrics objects.

        Returns:
            str: The LaTeX table code as a string.
        """
        if not runs:
            return ""

        run_names = list(runs.keys())
        all_metrics_data = {name: run.get_all_metrics() for name, run in runs.items()}
        
        table_rows = cls._get_table_rows()
        
        # Determine best values for each metric across all runs
        best_values = {}
        for metric_key in table_rows.keys():
            metric_values = []
            
            # Find the value for each run
            for run_data in all_metrics_data.values():
                val = None
                if metric_key in run_data["classification"]:
                    val = run_data["classification"][metric_key]
                elif metric_key in run_data["segmentation"]:
                    val = run_data["segmentation"][metric_key]
                metric_values.append(val)
            
            # Find the best value
            best_val = metric_values[0]
            for val in metric_values[1:]:
                if val is not None and cls._is_better(metric_key, val, best_val):
                    best_val = val
            best_values[metric_key] = best_val

        # Build the LaTeX string
        latex_str = "\\begin{table}[h!]\n"
        latex_str += "\\centering\n"
        latex_str += "\\begin{tabular}{|l|" + "l|" * len(run_names) + "}\n"
        latex_str += "\\hline\n"
        
        # Header row
        header = ["Metric"] + run_names
        latex_str += " & ".join(header) + " \\\\\n"
        latex_str += "\\hline\n"
        
        # Metric rows
        for metric_key, display_name in table_rows.items():
            row_values = []
            for run_name in run_names:
                run_data = all_metrics_data[run_name]
                val = None
                if metric_key in run_data["classification"]:
                    val = run_data["classification"][metric_key]
                elif metric_key in run_data["segmentation"]:
                    val = run_data["segmentation"][metric_key]
                
                # Format value and bold if it's the best
                formatted_val = f"${val:.4f}$" if isinstance(val, float) else f"${val}$"
                if val == best_values[metric_key]:
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
        run1 = Metrics.from_pickle(Path("results/run1_predictions.pkl"))
        run2 = Metrics.from_pickle(Path("results/run2_predictions.pkl"))
        
        # Save metrics and plots for a single run
        run1.save_all(output_dir="run1_output")
        print("Metrics and plots for Run 1 saved to run1_output/")

        # Compare multiple runs
        runs_to_compare = {
            "Run 1": run1,
            "Run 2": run2,
        }
        comparison_table = Metrics.compare_multiple_runs(runs_to_compare)
        print("\n--- Comparison LaTeX Table ---\n")
        print(comparison_table)
        
        # Save the comparison table to a file
        with open("comparison_table.tex", "w") as f:
            f.write(comparison_table)
        print("\nComparison table saved to comparison_table.tex")
        
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the prediction pickle files exist.")
