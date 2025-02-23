import os
import numpy as np
from math import sqrt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from anomalib.deploy import OpenVINOInferencer
from anomalib.data.utils import read_image
import matplotlib.pyplot as plt
from PIL import Image as im

tp, fp, tn, fn = 0, 0, 0, 0

# Define paths
dataset_path = "./datasets/PLMSTAR/soc/test"
openvino_model_path = "./weights/openvino/weights/openvino/model.bin"
metadata_path = "./weights/openvino/weights/openvino/metadata.json"

# Initialize inferencer
inferencer = OpenVINOInferencer(
    path=openvino_model_path,
    metadata=metadata_path
)

# Output directory for masks
output_directory = "./inf"
os.makedirs(output_directory, exist_ok=True)

# Function to process images in a directory
def process_images(base_path, label_type):
    pred_scores, pred_labels = [], []
    for class_dir in os.listdir(base_path):
        class_path = os.path.join(base_path, class_dir)
        if not os.path.isdir(class_path):
            continue  # Skip non-directory files
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            image = read_image(path=image_path)

            predictions = inferencer.predict(image=image)
            seg_image = im.fromarray(predictions.segmentations)
            seg_image.save(f"{output_directory}/{label_type}-seg-{class_dir}-{image_file}.png")

            pred_scores.append(predictions.pred_score)
            pred_labels.append(predictions.pred_label)
    return pred_scores, pred_labels

# Process normal (good) images
norm_path = os.path.join(dataset_path, "norm")
pred_score_good, pred_label_good = process_images(norm_path, label_type="good")

# Calculate TN and FN
tn = sum(1 for value in pred_label_good if not value)
fn = sum(1 for value in pred_label_good if value)

# Process anomalous (anom) images
anom_path = os.path.join(dataset_path, "anom")
pred_score_anom, pred_label_anom = process_images(anom_path, label_type="anom")

# Calculate TP and FP
tp = sum(1 for value in pred_label_anom if value)
fp = sum(1 for value in pred_label_anom if not value)

# Calculate metrics
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
g_mean = sqrt(recall * specificity)
mar = 1 - recall
far = 1 - specificity

# Print metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall/TPR/FDR/Sensitivity: {recall}")
print(f"F1 Score: {f1_score}")
print(f"Specificity: {specificity}")
print(f"G-mean: {g_mean}")
print(f"Missed Alarm Rate: {mar}")
print(f"False Alarm Rate: {far}")

# Generate ROC and PR curves
pred_scores = pred_score_good + pred_score_anom
true_labels = [0] * len(pred_score_good) + [1] * len(pred_score_anom)

fpr, tpr, thresholds = roc_curve(true_labels, pred_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')

precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
pr_auc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="upper right")
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.savefig('pr_curve.png')
