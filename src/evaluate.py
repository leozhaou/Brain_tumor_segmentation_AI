"""
Evaluation script for brain tumor segmentation.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import cv2
from pathlib import Path

from src.config import *
from src.data_loader import get_data_generators, get_image_mask_paths, load_image_and_mask
from src.model import dice_coefficient, dice_loss


def evaluate_model(model_path, batch_size=BATCH_SIZE, save_predictions=True):
    """
    Evaluate the trained model on the test set.

    Args:
        model_path (str): Path to the trained model
        batch_size (int): Batch size for evaluation
        save_predictions (bool): Whether to save prediction visualizations

    Returns:
        dict: Evaluation metrics
    """
    print("Starting model evaluation...")

    # Load model
    print(f"Loading model from {model_path}...")
    custom_objects = {
        'dice_coefficient': dice_coefficient,
        'dice_loss': dice_loss
    }
    model = load_model(model_path, custom_objects=custom_objects)
    print("Model loaded successfully")

    # Get data generators
    _, _, test_generator = get_data_generators(batch_size=batch_size)

    # Evaluate model
    print("Evaluating model on test set...")
    evaluation = model.evaluate(test_generator, verbose=1)

    # Print evaluation results
    metrics = dict(zip(model.metrics_names, evaluation))
    print("\nEvaluation metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    # Generate and evaluate predictions
    print("Generating predictions...")

    # Get a sample of test images and masks for visualization
    image_mask_pairs = get_image_mask_paths()
    _, _, test_pairs = split_dataset(image_mask_pairs)

    # Number of predictions to visualize
    num_visualize = min(10, len(test_pairs))

    # Lists to store metrics
    y_true_all = []
    y_pred_all = []
    dice_scores = []
    iou_scores = []

    # Create directory for prediction visualizations
    pred_vis_dir = os.path.join(PRED_DIR, 'visualizations')
    os.makedirs(pred_vis_dir, exist_ok=True)

    # Generate predictions and compute metrics
    for i, (image_path, mask_path) in enumerate(tqdm(test_pairs, desc="Evaluating predictions")):
        # Load image and mask
        image, mask = load_image_and_mask(image_path, mask_path, target_size=IMAGE_SIZE)

        # Expand dimensions to create a batch of size 1
        image_batch = np.expand_dims(image, axis=0)

        # Generate prediction
        prediction = model.predict(image_batch, verbose=0)[0]

        # Threshold prediction to get binary mask
        prediction_binary = (prediction > 0.5).astype(np.float32)

        # Flatten masks for metrics calculation
        mask_flat = mask.flatten()
        pred_flat = prediction_binary.flatten()

        # Append to lists for overall metrics
        y_true_all.extend(mask_flat)
        y_pred_all.extend(pred_flat)

        # Calculate Dice score for this sample
        dice_score = np.sum(2 * mask * prediction_binary) / (np.sum(mask) + np.sum(prediction_binary) + 1e-7)
        dice_scores.append(dice_score)

        # Calculate IoU (Intersection over Union) for this sample
        intersection = np.sum(mask * prediction_binary)
        union = np.sum(mask) + np.sum(prediction_binary) - intersection
        iou = intersection / (union + 1e-7)
        iou_scores.append(iou)

        # Save prediction visualization for a subset of samples
        if save_predictions and i < num_visualize:
            visualize_prediction(image, mask, prediction, prediction_binary,
                                 dice_score, iou, os.path.join(pred_vis_dir, f'prediction_{i + 1}.png'))

    # Calculate overall metrics
    print("\nCalculating overall metrics...")

    # Convert lists to numpy arrays
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    dice_scores = np.array(dice_scores)
    iou_scores = np.array(iou_scores)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1]).ravel()

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Print overall metrics
    print("\nOverall metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Mean Dice Coefficient: {np.mean(dice_scores):.4f} (±{np.std(dice_scores):.4f})")
    print(f"Mean IoU: {np.mean(iou_scores):.4f} (±{np.std(iou_scores):.4f})")

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'Specificity',
                   'F1 Score', 'Mean Dice Coefficient', 'Mean IoU'],
        'Value': [accuracy, precision, recall, specificity,
                  f1_score, np.mean(dice_scores), np.mean(iou_scores)]
    })
    metrics_csv_path = os.path.join(PRED_DIR, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Evaluation metrics saved to {metrics_csv_path}")

    # Create box plot of Dice scores and IoU scores
    plt.figure(figsize=(10, 6))
    plt.boxplot([dice_scores, iou_scores], labels=['Dice Coefficient', 'IoU'])
    plt.title('Distribution of Dice Coefficient and IoU Scores')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    box_plot_path = os.path.join(PRED_DIR, 'score_distribution.png')
    plt.savefig(box_plot_path)
    plt.close()
    print(f"Score distribution plot saved to {box_plot_path}")

    # Return metrics
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'dice_coefficient': np.mean(dice_scores),
        'iou': np.mean(iou_scores)
    }


def visualize_prediction(image, mask, prediction, prediction_binary, dice_score, iou, save_path):
    """
    Visualize the prediction and save it.

    Args:
        image: Input image
        mask: Ground truth mask
        prediction: Raw prediction (probability map)
        prediction_binary: Thresholded binary prediction
        dice_score: Dice coefficient for this sample
        iou: IoU for this sample
        save_path: Path to save the visualization
    """
    # Create figure
    plt.figure(figsize=(15, 10))

    # Plot input image
    plt.subplot(2, 3, 1)
    plt.title('Input Image (FLAIR Channel)')
    plt.imshow(image[:, :, 1], cmap='gray')  # Show FLAIR channel
    plt.axis('off')

    # Plot ground truth mask
    plt.subplot(2, 3, 2)
    plt.title('Ground Truth Mask')
    plt.imshow(mask[:, :, 0], cmap='binary')
    plt.axis('off')

    # Plot raw prediction
    plt.subplot(2, 3, 3)
    plt.title('Raw Prediction (Probability Map)')
    plt.imshow(prediction[:, :, 0], cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    # Plot binary prediction
    plt.subplot(2, 3, 4)
    plt.title('Binary Prediction (Threshold = 0.5)')
    plt.imshow(prediction_binary[:, :, 0], cmap='binary')
    plt.axis('off')

    # Plot overlay of prediction on image
    plt.subplot(2, 3, 5)
    plt.title('Overlay (Ground Truth)')
    overlay_gt = create_overlay(image[:, :, 1], mask[:, :, 0], color=[0, 1, 0])
    plt.imshow(overlay_gt)
    plt.axis('off')

    # Plot overlay of prediction on image
    plt.subplot(2, 3, 6)
    plt.title('Overlay (Prediction)')
    overlay_pred = create_overlay(image[:, :, 1], prediction_binary[:, :, 0], color=[1, 0, 0])
    plt.imshow(overlay_pred)
    plt.axis('off')

    # Add metrics as subtitle
    plt.suptitle(f'Dice Score: {dice_score:.4f}, IoU: {iou:.4f}', fontsize=16)

    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def create_overlay(image, mask, color=[1, 0, 0]):
    """
    Create an overlay of the mask on the image.

    Args:
        image: Grayscale image
        mask: Binary mask
        color: RGB color for the mask overlay

    Returns:
        Overlay image
    """
    # Normalize image to 0-1 if not already
    if image.max() > 1:
        image = image / 255.0

    # Create RGB image
    img_rgb = np.stack([image] * 3, axis=2)

    # Create mask color overlay
    mask_rgb = np.zeros_like(img_rgb)
    for i in range(3):
        mask_rgb[:, :, i] = mask * color[i]

    # Blend image and mask
    alpha = 0.5
    overlay = img_rgb * (1 - alpha * mask[:, :, np.newaxis]) + mask_rgb * alpha

    return overlay


def split_dataset(image_mask_pairs, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO,
                  random_seed=RANDOM_SEED):
    """
    Split the dataset into training, validation, and test sets.

    Args:
        image_mask_pairs (list): List of tuples (image_path, mask_path)
        train_ratio (float): Proportion of data for training
        val_ratio (float): Proportion of data for validation
        test_ratio (float): Proportion of data for testing
        random_seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_pairs, val_pairs, test_pairs)
    """
    # Shuffle image_mask_pairs
    np.random.seed(random_seed)
    np.random.shuffle(image_mask_pairs)

    # Calculate split indices
    n = len(image_mask_pairs)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # Split the data
    train_pairs = image_mask_pairs[:train_end]
    val_pairs = image_mask_pairs[train_end:val_end]
    test_pairs = image_mask_pairs[val_end:]

    return train_pairs, val_pairs, test_pairs


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate U-Net model for brain tumor segmentation')
    parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_DIR, 'unet_brain_tumor_best.h5'),
                        help='Path to the trained model')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for evaluation')
    parser.add_argument('--no_save_predictions', action='store_true', help='Do not save prediction visualizations')

    args = parser.parse_args()

    # Create output directories if they don't exist
    os.makedirs(PRED_DIR, exist_ok=True)

    # Evaluate model
    evaluate_model(
        model_path=args.model_path,
        batch_size=args.batch_size,
        save_predictions=not args.no_save_predictions
    )