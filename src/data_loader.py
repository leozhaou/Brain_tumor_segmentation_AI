"""
Data loading and preprocessing for brain tumor segmentation.
"""
import os
import glob
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from PIL import Image
import random
from src.config import *


def load_metadata():
    """
    Load metadata from data.csv file.
    """
    csv_path = os.path.join(DATA_DIR, 'data.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        print(f"Warning: data.csv not found at {csv_path}")
        return None


def get_image_mask_paths():
    """
    Get paths to all images and their corresponding masks.
    Returns:
        list: List of tuples (image_path, mask_path)
    """
    # Get all patient folders
    patient_folders = [f for f in glob.glob(os.path.join(DATA_DIR, '*')) if os.path.isdir(f)]

    # Get all image and mask pairs
    image_mask_pairs = []
    for folder in patient_folders:
        # Get all .tif images that don't have "_mask" in their name
        images = glob.glob(os.path.join(folder, "*.tif"))
        images = [img for img in images if "_mask" not in img]

        for img_path in images:
            # Construct mask path
            mask_path = img_path.replace('.tif', '_mask.tif')

            # Check if both image and mask exist
            if os.path.exists(mask_path):
                image_mask_pairs.append((img_path, mask_path))

    return image_mask_pairs


def load_image_and_mask(image_path, mask_path, target_size=IMAGE_SIZE):
    # Load image
    image = np.array(Image.open(image_path))
    # Load mask
    mask = np.array(Image.open(mask_path))
    # Convert to grayscale if mask has more than one channel
    if len(mask.shape) > 2 and mask.shape[2] > 1:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    # Resize both image and mask
    image = cv2.resize(image, target_size[::-1])  # OpenCV uses (width, height)
    mask = cv2.resize(mask, target_size[::-1], interpolation=cv2.INTER_NEAREST)
    # Normalize image
    image = image / 255.0
    # Binarize mask (threshold at 0.5)
    mask = (mask > 0).astype(np.float32)
    # Reshape mask to (height, width, 1)
    mask = np.expand_dims(mask, axis=-1)
    return image, mask


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
    # First split: separate training set
    train_pairs, temp_pairs = train_test_split(
        image_mask_pairs,
        train_size=train_ratio,
        random_state=random_seed
    )

    # Second split: separate validation and test sets
    val_size = val_ratio / (val_ratio + test_ratio)
    val_pairs, test_pairs = train_test_split(
        temp_pairs,
        train_size=val_size,
        random_state=random_seed
    )

    print(f"Dataset split: {len(train_pairs)} training, {len(val_pairs)} validation, {len(test_pairs)} test samples")

    return train_pairs, val_pairs, test_pairs


class BrainTumorDataGenerator(Sequence):
    """
    Data generator for brain tumor segmentation.
    """

    def __init__(self, image_mask_pairs, batch_size=BATCH_SIZE, target_size=IMAGE_SIZE, augment=False, shuffle=True):
        """
        Initialize the data generator.

        Args:
            image_mask_pairs (list): List of tuples (image_path, mask_path)
            batch_size (int): Batch size
            target_size (tuple): Target size for resizing (height, width)
            augment (bool): Whether to apply data augmentation
            shuffle (bool): Whether to shuffle the data after each epoch
        """
        self.image_mask_pairs = image_mask_pairs
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.shuffle = shuffle

        self.on_epoch_end()

        # Setup data augmentation
        if self.augment:
            self.image_datagen = ImageDataGenerator(
                rotation_range=ROTATION_RANGE,
                width_shift_range=WIDTH_SHIFT_RANGE,
                height_shift_range=HEIGHT_SHIFT_RANGE,
                shear_range=SHEAR_RANGE,
                zoom_range=ZOOM_RANGE,
                horizontal_flip=HORIZONTAL_FLIP,
                vertical_flip=VERTICAL_FLIP,
                fill_mode=FILL_MODE
            )
            self.mask_datagen = ImageDataGenerator(
                rotation_range=ROTATION_RANGE,
                width_shift_range=WIDTH_SHIFT_RANGE,
                height_shift_range=HEIGHT_SHIFT_RANGE,
                shear_range=SHEAR_RANGE,
                zoom_range=ZOOM_RANGE,
                horizontal_flip=HORIZONTAL_FLIP,
                vertical_flip=VERTICAL_FLIP,
                fill_mode=FILL_MODE
            )

            # Same seed for both generators
            self.seed = random.randint(1, 1000)

    def __len__(self):
        """
        Return the number of batches per epoch.
        """
        return int(np.ceil(len(self.image_mask_pairs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data.

        Args:
            index (int): Batch index

        Returns:
            tuple: (batch_images, batch_masks)
        """
        # Generate indexes for this batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Find the image-mask pairs for this batch
        batch_pairs = [self.image_mask_pairs[i] for i in batch_indices]

        # Generate batch of images and masks
        X, y = self._generate_batch(batch_pairs)

        return X, y

    def on_epoch_end(self):
        """
        Updates indices after each epoch.
        """
        self.indices = np.arange(len(self.image_mask_pairs))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _generate_batch(self, batch_pairs):
        """
        Generate a batch of images and masks.

        Args:
            batch_pairs (list): List of tuples (image_path, mask_path) for this batch

        Returns:
            tuple: (batch_images, batch_masks)
        """
        # Initialize batch arrays
        batch_images = np.empty((len(batch_pairs), *self.target_size, CHANNELS))
        batch_masks = np.empty((len(batch_pairs), *self.target_size, MASK_CHANNELS))

        # Load images and masks for this batch
        for i, (image_path, mask_path) in enumerate(batch_pairs):
            image, mask = load_image_and_mask(image_path, mask_path, self.target_size)

            # Apply augmentation if necessary
            if self.augment:
                # Apply the same transformation to image and mask
                transform_params = self.image_datagen.get_random_transform(image.shape)

                image = self.image_datagen.apply_transform(image, transform_params)
                mask = self.mask_datagen.apply_transform(mask, transform_params)

                # Ensure mask values remain binary after augmentation
                mask = (mask > 0.5).astype(np.float32)

            batch_images[i] = image
            batch_masks[i] = mask

        return batch_images, batch_masks


def get_data_generators(batch_size=BATCH_SIZE, target_size=IMAGE_SIZE):
    """
    Get data generators for training, validation, and testing.

    Args:
        batch_size (int): Batch size
        target_size (tuple): Target size for resizing (height, width)

    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    # Get all image and mask pairs
    image_mask_pairs = get_image_mask_paths()

    if not image_mask_pairs:
        raise ValueError("No image-mask pairs found. Check your data directory.")

    # Split dataset
    train_pairs, val_pairs, test_pairs = split_dataset(image_mask_pairs)

    # Create data generators
    train_generator = BrainTumorDataGenerator(
        train_pairs,
        batch_size=batch_size,
        target_size=target_size,
        augment=AUGMENTATION,
        shuffle=True
    )

    val_generator = BrainTumorDataGenerator(
        val_pairs,
        batch_size=batch_size,
        target_size=target_size,
        augment=False,
        shuffle=False
    )

    test_generator = BrainTumorDataGenerator(
        test_pairs,
        batch_size=batch_size,
        target_size=target_size,
        augment=False,
        shuffle=False
    )

    return train_generator, val_generator, test_generator


if __name__ == "__main__":
    # Test the data loader
    print("Testing data loader...")

    # Load metadata
    metadata = load_metadata()
    if metadata is not None:
        print(f"Loaded metadata with {len(metadata)} patients")

    # Get all image and mask pairs
    image_mask_pairs = get_image_mask_paths()
    print(f"Found {len(image_mask_pairs)} image-mask pairs")

    if len(image_mask_pairs) > 0:
        # Test loading an image and mask
        image_path, mask_path = image_mask_pairs[0]
        print(f"Testing with image: {image_path}")
        print(f"Testing with mask: {mask_path}")

        image, mask = load_image_and_mask(image_path, mask_path)
        print(f"Loaded image with shape: {image.shape}")
        print(f"Loaded mask with shape: {mask.shape}")

        # Test data generators
        train_generator, val_generator, test_generator = get_data_generators()

        print(f"Train generator has {len(train_generator)} batches")
        print(f"Validation generator has {len(val_generator)} batches")
        print(f"Test generator has {len(test_generator)} batches")

        # Test getting a batch
        batch_images, batch_masks = train_generator[0]
        print(f"Batch of images has shape: {batch_images.shape}")
        print(f"Batch of masks has shape: {batch_masks.shape}")

        print("Data loader test completed successfully")
    else:
        print("No image-mask pairs found. Check your data directory.")