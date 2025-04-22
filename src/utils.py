"""
Utility functions for brain tumor segmentation.
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import KFold
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

from src.config import *


def create_dir_if_not_exists(directory):
    """
    Create a directory if it doesn't exist.

    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def get_class_weights(train_generator):
    """
    Calculate class weights for imbalanced dataset.

    Args:
        train_generator: Training data generator

    Returns:
        dict: Class weights dictionary
    """
    # Get all masks from the generator
    y_true = []
    for _, y in tqdm(train_generator, desc="Calculating class weights"):
        y_true.append(y)

    y_true = np.concatenate(y_true, axis=0)
    y_true = y_true.flatten()

    # Count class occurrences
    negative_count = np.sum(y_true == 0)
    positive_count = np.sum(y_true == 1)
    total_count = negative_count + positive_count

    # Calculate class weights
    class_weights = {
        0: total_count / (2 * negative_count),
        1: total_count / (2 * positive_count)
    }

    print(f"Class weights: {class_weights}")
    print(
        f"Class distribution: {positive_count / total_count:.2%} positive, {negative_count / total_count:.2%} negative")

    return class_weights


def predict_and_visualize(model, image_path, mask_path=None, save_path=None):
    """
    Make a prediction on a single image and visualize the result.

    Args:
        model: Trained model
        image_path (str): Path to the input image
        mask_path (str, optional): Path to the ground truth mask
        save_path (str, optional): Path to save the visualization

    Returns:
        Prediction binary mask
    """
    # Load image
    image = np.array(Image.open(image_path))

    # Resize image
    image = cv2.resize(image, IMAGE_SIZE[::-1])  # OpenCV uses (width, height)

    # Normalize image
    image = image / 255.0

    # Expand dimensions to create a batch of size 1
    image_batch = np.expand_dims(image, axis=0)

    # Generate prediction
    prediction = model.predict(image_batch)[0]

    # Threshold prediction to get binary mask
    prediction_binary = (prediction > 0.5).astype(np.float32)

    # Create figure
    plt.figure(figsize=(15, 10))

    # Plot input image
    plt.subplot(2, 3, 1)
    plt.title('Input Image (FLAIR Channel)')
    plt.imshow(image[:, :, 1], cmap='gray')  # Show FLAIR channel
    plt.axis('off')

    # Plot raw prediction
    plt.subplot(2, 3, 2)
    plt.title('Raw Prediction (Probability Map)')
    plt.imshow(prediction[:, :, 0], cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    # Plot binary prediction
    plt.subplot(2, 3, 3)
    plt.title('Binary Prediction (Threshold = 0.5)')
    plt.imshow(prediction_binary[:, :, 0], cmap='binary')
    plt.axis('off')

    # Plot overlay of prediction on image
    plt.subplot(2, 3, 4)
    plt.title('Overlay (Prediction)')
    overlay_pred = create_overlay(image[:, :, 1], prediction_binary[:, :, 0], color=[1, 0, 0])
    plt.imshow(overlay_pred)
    plt.axis('off')

    # If ground truth mask is provided
    if mask_path and os.path.exists(mask_path):
        # Load mask
        mask = np.array(Image.open(mask_path))

        # Resize mask
        mask = cv2.resize(mask, IMAGE_SIZE[::-1], interpolation=cv2.INTER_NEAREST)

        # Binarize mask
        mask = (mask > 0).astype(np.float32)

        # If mask has more than one channel, convert to grayscale
        if len(mask.shape) > 2 and mask.shape[2] > 1:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        # Plot ground truth mask
        plt.subplot(2, 3, 5)
        plt.title('Ground Truth Mask')
        plt.imshow(mask, cmap='binary')
        plt.axis('off')

        # Plot overlay of ground truth on image
        plt.subplot(2, 3, 6)
        plt.title('Overlay (Ground Truth)')
        overlay_gt = create_overlay(image[:, :, 1], mask, color=[0, 1, 0])
        plt.imshow(overlay_gt)
        plt.axis('off')

        # Calculate metrics
        dice_score = np.sum(2 * mask * prediction_binary[:, :, 0]) / (
                    np.sum(mask) + np.sum(prediction_binary[:, :, 0]) + 1e-7)
        intersection = np.sum(mask * prediction_binary[:, :, 0])
        union = np.sum(mask) + np.sum(prediction_binary[:, :, 0]) - intersection
        iou = intersection / (union + 1e-7)

        # Add metrics as subtitle
        plt.suptitle(f'Dice Score: {dice_score:.4f}, IoU: {iou:.4f}', fontsize=16)

    # Adjust layout
    plt.tight_layout()

    # Save or show visualization
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

    # Close figure
    plt.close()

    return prediction_binary


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


def setup_k_fold_cross_validation(image_mask_pairs, n_splits=5, random_seed=RANDOM_SEED):
    """
    Setup k-fold cross-validation.

    Args:
        image_mask_pairs (list): List of tuples (image_path, mask_path)
        n_splits (int): Number of folds
        random_seed (int): Random seed for reproducibility

    Returns:
        list: List of tuples (train_pairs, val_pairs) for each fold
    """
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    # Create folds
    folds = []
    for train_indices, val_indices in kf.split(image_mask_pairs):
        train_pairs = [image_mask_pairs[i] for i in train_indices]
        val_pairs = [image_mask_pairs[i] for i in val_indices]
        folds.append((train_pairs, val_pairs))

    return folds


def apply_preprocessing(image, preprocessing_steps):
    """
    Apply preprocessing steps to an image.

    Args:
        image: Input image
        preprocessing_steps (list): List of preprocessing steps to apply

    Returns:
        Preprocessed image
    """
    result = image.copy()

    for step in preprocessing_steps:
        if step == 'clahe':
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            if len(result.shape) == 3:
                for i in range(result.shape[2]):
                    # Create CLAHE object
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    # Apply CLAHE to each channel
                    result[:, :, i] = clahe.apply(np.uint8(result[:, :, i] * 255)) / 255.0
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                result = clahe.apply(np.uint8(result * 255)) / 255.0

        elif step == 'gaussian_blur':
            # Apply Gaussian blur
            result = cv2.GaussianBlur(result, (5, 5), 0)

        elif step == 'histogram_equalization':
            # Apply histogram equalization
            if len(result.shape) == 3:
                # Convert to YUV color space
                yuv = cv2.cvtColor(result, cv2.COLOR_RGB2YUV)
                # Apply histogram equalization to the Y channel
                yuv[:, :, 0] = cv2.equalizeHist(np.uint8(yuv[:, :, 0] * 255)) / 255.0
                # Convert back to RGB
                result = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
            else:
                result = cv2.equalizeHist(np.uint8(result * 255)) / 255.0

        elif step == 'normalize':
            # Normalize to zero mean and unit variance
            mean = np.mean(result)
            std = np.std(result)
            result = (result - mean) / (std + 1e-7)

    return result


def augment_data(image, mask, augmentation_params=None):
    """
    Apply data augmentation to an image and its mask.

    Args:
        image: Input image
        mask: Input mask
        augmentation_params (dict): Augmentation parameters

    Returns:
        tuple: (augmented_image, augmented_mask)
    """
    if augmentation_params is None:
        augmentation_params = {
            'rotation_range': ROTATION_RANGE,
            'width_shift_range': WIDTH_SHIFT_RANGE,
            'height_shift_range': HEIGHT_SHIFT_RANGE,
            'shear_range': SHEAR_RANGE,
            'zoom_range': ZOOM_RANGE,
            'horizontal_flip': HORIZONTAL_FLIP,
            'vertical_flip': VERTICAL_FLIP,
            'fill_mode': FILL_MODE
        }

    # Create a random seed for this augmentation
    seed = np.random.randint(1, 1000)

    # Create data generators
    image_datagen = ImageDataGenerator(**augmentation_params)
    mask_datagen = ImageDataGenerator(**augmentation_params)

    # Reshape for data generators (add batch dimension)
    image_batch = np.expand_dims(image, axis=0)
    mask_batch = np.expand_dims(mask, axis=0)

    # Apply the same transformation to image and mask
    image_gen = image_datagen.flow(image_batch, batch_size=1, seed=seed)
    mask_gen = mask_datagen.flow(mask_batch, batch_size=1, seed=seed)

    # Get augmented image and mask
    augmented_image = image_gen.next()[0]
    augmented_mask = mask_gen.next()[0]

    # Ensure mask values remain binary after augmentation
    augmented_mask = (augmented_mask > 0.5).astype(np.float32)

    return augmented_image, augmented_mask


def ensemble_predictions(models, image, threshold=0.5):
    """
    Combine predictions from multiple models.

    Args:
        models (list): List of trained models
        image: Input image
        threshold (float): Threshold for binary prediction

    Returns:
        tuple: (ensemble_prediction, ensemble_binary)
    """
    # Expand dimensions to create a batch of size 1
    image_batch = np.expand_dims(image, axis=0)

    # Initialize ensemble prediction
    ensemble_prediction = None

    # Get predictions from all models
    for model in models:
        prediction = model.predict(image_batch)[0]

        if ensemble_prediction is None:
            ensemble_prediction = prediction
        else:
            ensemble_prediction += prediction

    # Average predictions
    ensemble_prediction /= len(models)

    # Threshold prediction to get binary mask
    ensemble_binary = (ensemble_prediction > threshold).astype(np.float32)

    return ensemble_prediction, ensemble_binary


def visualize_dataset_samples(image_mask_pairs, num_samples=5, save_dir=None):
    """
    Visualize random samples from the dataset.

    Args:
        image_mask_pairs (list): List of tuples (image_path, mask_path)
        num_samples (int): Number of samples to visualize
        save_dir (str, optional): Directory to save visualizations
    """
    # Select random samples
    np.random.seed(RANDOM_SEED)
    indices = np.random.choice(len(image_mask_pairs), size=min(num_samples, len(image_mask_pairs)), replace=False)

    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Visualize each sample
    for i, idx in enumerate(indices):
        image_path, mask_path = image_mask_pairs[idx]

        # Load image and mask
        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path))

        # Convert to grayscale if mask has more than one channel
        if len(mask.shape) > 2 and mask.shape[2] > 1:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        # Binarize mask
        mask = (mask > 0).astype(np.float32)

        # Create figure
        plt.figure(figsize=(15, 5))

        # Plot input image (all channels)
        plt.subplot(1, 5, 1)
        plt.title('Original Image')
        plt.imshow(image)
        plt.axis('off')

        # Plot FLAIR channel
        plt.subplot(1, 5, 2)
        plt.title('FLAIR Channel')
        plt.imshow(image[:, :, 1], cmap='gray')
        plt.axis('off')

        # Plot ground truth mask
        plt.subplot(1, 5, 3)
        plt.title('Ground Truth Mask')
        plt.imshow(mask, cmap='binary')
        plt.axis('off')

        # Plot overlay of mask on image
        plt.subplot(1, 5, 4)
        plt.title('Overlay')
        overlay = create_overlay(image[:, :, 1], mask, color=[0, 1, 0])
        plt.imshow(overlay)
        plt.axis('off')

        # Plot 3D channel view
        plt.subplot(1, 5, 5)
        plt.title('Channel Comparison')
        channel_vis = np.zeros_like(image)
        channel_vis[:, :, 0] = image[:, :, 0]  # Pre-contrast in Red
        channel_vis[:, :, 1] = image[:, :, 1]  # FLAIR in Green
        channel_vis[:, :, 2] = image[:, :, 2]  # Post-contrast in Blue
        plt.imshow(channel_vis)
        plt.axis('off')

        # Add filename as suptitle
        plt.suptitle(f'Sample {i + 1}: {os.path.basename(image_path)}', fontsize=16)

        # Adjust layout
        plt.tight_layout()

        # Save or show visualization
        if save_dir:
            save_path = os.path.join(save_dir, f'sample_{i + 1}.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()

        # Close figure
        plt.close()


def get_model_size_and_params(model):
    """
    Get model size and number of parameters.

    Args:
        model: Keras model

    Returns:
        tuple: (model_size_mb, num_parameters)
    """
    # Count parameters
    num_parameters = model.count_params()

    # Estimate model size in MB
    model_size_mb = num_parameters * 4 / (1024 * 1024)  # Assuming 4 bytes per parameter

    return model_size_mb, num_parameters


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")

    # Create directories
    for directory in [OUTPUT_DIR, MODEL_DIR, LOG_DIR, PRED_DIR]:
        create_dir_if_not_exists(directory)

    print("Utility functions test completed successfully")