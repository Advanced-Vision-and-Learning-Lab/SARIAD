import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from anomalib.deploy import OpenVINOInferencer
from anomalib.data.utils import read_image
from PIL import Image as im
import cv2
from math import sqrt

tp, fp, tn, fn = 0, 0, 0, 0

# Define paths
dataset_path = "./datasets/PLMSTAR/soc/test"
openvino_model_path = "./weights/openvino/weights/openvino/model.bin"
metadata_path = "./weights/openvino/weights/openvino/metadata.json"

assert os.path.exists(openvino_model_path), "Model binary file not found!"
assert os.path.exists(metadata_path), "Metadata file not found!"

# Initialize inferencer
inferencer = OpenVINOInferencer(
    path=openvino_model_path,
    metadata=metadata_path,
)

# Output directory for masks
output_directory = "./inf"
os.makedirs(output_directory, exist_ok=True)

def process_images(base_path, label_type):
    pred_scores, pred_labels = [], []
    
    # Check if the base_path itself contains images (specific target case)
    if all(file.endswith(('.jpg', '.png', '.jpeg')) for file in os.listdir(base_path)):
        for image_file in os.listdir(base_path):
            image_path = os.path.join(base_path, image_file)
            
            # Read and preprocess the image
            image = read_image(path=image_path)
            
            # Predict using the inferencer
            predictions = inferencer.predict(image=image)
            
            # Append prediction results
            pred_scores.append(predictions.pred_score)
            pred_labels.append(predictions.pred_label)
    else:
        # Handle case where base_path contains subdirectories (entire dataset case)
        for class_dir in os.listdir(base_path):
            class_path = os.path.join(base_path, class_dir)
            
            # Skip non-directory files
            if not os.path.isdir(class_path):
                continue
                
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                
                # Read and preprocess the image
                image = read_image(path=image_path)
                
                # Predict using the inferencer
                predictions = inferencer.predict(image=image)
                
                # Append prediction results
                pred_scores.append(predictions.pred_score)
                pred_labels.append(predictions.pred_label)
    
    return pred_scores, pred_labels

def calculate_metrics(pred_scores, pred_labels_good, pred_labels_anom, target_name="all_targets"):
    # Ensure binary labels for good (0) and anom (1)
    true_labels = [0] * len(pred_labels_good) + [1] * len(pred_labels_anom)

    # Calculate confusion matrix components (TP, TN, FP, FN)
    tp = sum(1 for value in pred_labels_anom if value == 1)
    tn = sum(1 for value in pred_labels_good if value == 0)
    fp = sum(1 for value in pred_labels_anom if value == 0)
    fn = sum(1 for value in pred_labels_good if value == 1)

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    g_mean = sqrt(recall * specificity)
    mar = 1 - recall
    far = 1 - specificity

    with open("./results/results.txt", "a") as file:
        file.write(f"Results for Target: {target_name}\n")
        file.write(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall/TPR/FDR/Sensitivity: {recall:.4f}\n")
        file.write(f"F1 Score: {f1_score:.4f}\n")
        file.write(f"Specificity: {specificity:.4f}\n")
        file.write(f"G-mean: {g_mean:.4f}\n")
        file.write(f"Missed Alarm Rate: {mar:.4f}\n")
        file.write(f"False Alarm Rate: {far:.4f}\n")
        file.write("\n")

    # Print metrics
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall/TPR/FDR/Sensitivity: {recall}")
    print(f"F1 Score: {f1_score}")
    print(f"Specificity: {specificity}")
    print(f"G-mean: {g_mean}")
    print(f"Missed Alarm Rate: {mar}")
    print(f"False Alarm Rate: {far}")

    # Generate ROC and PR curves
    # Convert pred_labels_good and pred_labels_anom to binary labels (0 for good, 1 for anom)
    fpr, tpr, thresholds = roc_curve(true_labels, pred_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve for {target_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'./results/roc_curve_{target_name}.png')

    # PR Curve
    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall curve for {target_name}')
    plt.legend(loc="upper right")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.savefig(f'./results/pr_curve_{target_name}.png')

# Process all targets first
print(dataset_path)
norm_path = os.path.join(dataset_path, "norm")
pred_score_good, pred_label_good = process_images(norm_path, label_type="good")

# Process anomalous (anom) images
anom_path = os.path.join(dataset_path, "anom")
pred_score_anom, pred_label_anom = process_images(anom_path, label_type="anom")

# Calculate and plot for all targets
calculate_metrics(pred_score_good + pred_score_anom, pred_label_good, pred_label_anom, target_name="all_targets")

# Now process for each target filter
target_filters = ["2S1","BMP2","BRDM2","BTR60","BTR70","D7","T62","T72","ZIL131","ZSU234"] 
for target_filter in target_filters:
    print(target_filter)
    norm_path_filtered = os.path.join(dataset_path, "norm", target_filter)
    print(f"norm_path_filtered: {norm_path_filtered}")
    pred_score_good_filtered, pred_label_good_filtered = process_images(norm_path_filtered, label_type="good")

    anom_path_filtered = os.path.join(dataset_path, "anom", target_filter)
    print(f"anom_path_filtered: {anom_path_filtered}")
    pred_score_anom_filtered, pred_label_anom_filtered = process_images(anom_path_filtered, label_type="anom")

    # Calculate and plot for each target
    calculate_metrics(pred_score_good_filtered + pred_score_anom_filtered, 
                      pred_label_good_filtered, 
                      pred_label_anom_filtered, 
                      target_name=target_filter)
