"""
Configuration parameters for the brain tumor segmentation project.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
PRED_DIR = os.path.join(OUTPUT_DIR, 'predictions')

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, MODEL_DIR, LOG_DIR, PRED_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data parameters
IMAGE_SIZE = (256, 256)  # Resize images to this size
CHANNELS = 3  # Number of channels in the input image
MASK_CHANNELS = 1  # Number of channels in the mask
TRAIN_RATIO = 0.8  # Percentage of data used for training
VAL_RATIO = 0.1  # Percentage of data used for validation
TEST_RATIO = 0.1  # Percentage of data used for testing
RANDOM_SEED = 42  # For reproducibility

# Model parameters
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.3
FILTERS_BASE = 64  # Number of filters in the first layer (doubles with each downsampling)
LOSS_FUNCTION = 'binary_crossentropy'  # or 'dice_loss'
METRICS = ['accuracy', 'dice_coefficient']  # Metrics to track during training

# Augmentation parameters
AUGMENTATION = True
ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
SHEAR_RANGE = 0.1
ZOOM_RANGE = 0.1
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
FILL_MODE = 'nearest'