import os
import numpy as np
from PIL import Image
from collections import defaultdict

def count_predictions(folder, num_classes):
    """
    Count predictions for each class in the GT against the predictions.
    Also count the GT class occurrences for each predicted class.

    :param folder: Directory containing both GT and prediction images.
    :param num_classes: Number of classes in the segmentation.
    :return: Two dictionaries with counts: 
             1. counts_gt_to_pred - counts of predictions for each GT class.
             2. counts_pred_to_gt - counts of GT class occurrences for each predicted class.
    """
    # Initialize dictionaries to hold the counts
    counts_gt_to_pred = {i: defaultdict(int) for i in range(num_classes)}
    counts_pred_to_gt = {i: defaultdict(int) for i in range(num_classes)}

    # List files in the directory
    files = os.listdir(folder)

    for file in files:
        if file.endswith('_gt.png'):
            gt_file = file
            pred_file = file.replace('_gt.png', '.png')
            gt_path = os.path.join(folder, gt_file)
            pred_path = os.path.join(folder, pred_file)

            if not os.path.exists(pred_path):
                print(f"Prediction file {pred_path} not found, skipping.")
                continue

            gt_img = np.array(Image.open(gt_path))
            pred_img = np.array(Image.open(pred_path))

            for gt_class in range(num_classes):
                # Mask for the current ground truth class
                gt_mask = (gt_img == gt_class)
                # Count occurrences of each predicted class in the masked area
                for pred_class in range(num_classes):
                    counts_gt_to_pred[gt_class][pred_class] += np.sum(pred_img[gt_mask] == pred_class)

            for pred_class in range(num_classes):
                # Mask for the current predicted class
                pred_mask = (pred_img == pred_class)
                # Count occurrences of each GT class in the masked area
                for gt_class in range(num_classes):
                    counts_pred_to_gt[pred_class][gt_class] += np.sum(gt_img[pred_mask] == gt_class)

    return counts_gt_to_pred, counts_pred_to_gt

# Set the directory and number of classes
folder = 'path/to/folder'
num_classes = 5  # Replace with the actual number of classes

# Get the counts
counts_gt_to_pred, counts_pred_to_gt = count_predictions(folder, num_classes)

# Print the results
print("Counts of predictions for each GT class:")
for gt_class, pred_counts in counts_gt_to_pred.items():
    print(f"GT Class {gt_class}:")
    for pred_class, count in pred_counts.items():
        print(f"  Predicted Class {pred_class}: {count}")

print("\nCounts of GT class occurrences for each predicted class:")
for pred_class, gt_counts in counts_pred_to_gt.items():
    print(f"Predicted Class {pred_class}:")
    for gt_class, count in gt_counts.items():
        print(f"  GT Class {gt_class}: {count}")
