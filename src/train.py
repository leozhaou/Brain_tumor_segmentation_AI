"""
Training script for brain tumor segmentation.
"""
import os
import argparse
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt

from src.config import *
from src.data_loader import get_data_generators
from src.model import build_unet, compile_model, get_model_callbacks, dice_coefficient, dice_loss


def train_model(batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE,
                loss_function=LOSS_FUNCTION, model_path=None, continue_training=False):
    """
    Train the U-Net model for brain tumor segmentation.

    Args:
        batch_size (int): Batch size
        epochs (int): Number of epochs
        learning_rate (float): Learning rate
        loss_function (str): Loss function name ('binary_crossentropy' or 'dice_loss')
        model_path (str): Path to a pre-trained model to continue training
        continue_training (bool): Whether to continue training a pre-trained model

    Returns:
        tuple: (trained_model, history)
    """
    print("Starting model training...")

    # Get data generators
    print("Preparing data generators...")
    train_generator, val_generator, _ = get_data_generators(batch_size=batch_size)

    # Build or load model
    if continue_training and model_path:
        print(f"Loading pre-trained model from {model_path}...")
        # Define custom objects for the custom loss and metrics
        custom_objects = {
            'dice_coefficient': dice_coefficient,
            'dice_loss': dice_loss
        }
        model = load_model(model_path, custom_objects=custom_objects)
        print("Pre-trained model loaded successfully")
    else:
        print("Building U-Net model...")
        input_shape = (*IMAGE_SIZE, CHANNELS)
        model = build_unet(input_shape=input_shape, filters_base=FILTERS_BASE, dropout_rate=DROPOUT_RATE)
        model = compile_model(model, learning_rate=learning_rate, loss_function=loss_function)
        print("Model built successfully")

    # Print model summary
    model.summary()

    # Try to plot model architecture
    try:
        model_plot_path = os.path.join(MODEL_DIR, 'unet_architecture.png')
        plot_model(model, to_file=model_plot_path, show_shapes=True, show_layer_names=True)
        print(f"Model architecture plotted to {model_plot_path}")
    except ImportError:
        print("Warning: Could not plot model architecture. Graphviz is not installed.")
        print("To install Graphviz, visit https://graphviz.gitlab.io/download/")
        print("Training will continue without creating the model diagram.")
    except Exception as e:
        print(f"Warning: Failed to plot model architecture: {e}")
        print("Training will continue without creating the model diagram.")

    # Get callbacks
    callbacks = get_model_callbacks()

    # Train model
    print(f"Training model for {epochs} epochs...")
    start_time = time.time()

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Model training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    # Save final model
    final_model_path = os.path.join(MODEL_DIR, 'unet_brain_tumor_final.h5')
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Plot training history
    plot_training_history(history)

    return model, history


def plot_training_history(history):
    """
    Plot training history.

    Args:
        history: Training history object
    """
    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot training & validation accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    # Plot training & validation loss
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    # Plot training & validation dice coefficient
    plt.subplot(2, 2, 3)
    plt.plot(history.history['dice_coefficient'])
    plt.plot(history.history['val_dice_coefficient'])
    plt.title('Dice Coefficient')
    plt.ylabel('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    # Plot learning rate if available
    if 'lr' in history.history:
        plt.subplot(2, 2, 4)
        plt.plot(history.history['lr'])
        plt.title('Learning Rate')
        plt.ylabel('Learning Rate')
        plt.xlabel('Epoch')
        plt.yscale('log')

    # Save plot
    history_plot_path = os.path.join(LOG_DIR, 'training_history.png')
    plt.tight_layout()
    plt.savefig(history_plot_path)
    print(f"Training history plotted to {history_plot_path}")

    # Close plot
    plt.close()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train U-Net model for brain tumor segmentation')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--loss', type=str, default=LOSS_FUNCTION, choices=['binary_crossentropy', 'dice_loss'],
                        help='Loss function')
    parser.add_argument('--model_path', type=str, default=None, help='Path to a pre-trained model to continue training')
    parser.add_argument('--continue_training', action='store_true', help='Continue training a pre-trained model')

    args = parser.parse_args()

    # Create output directories if they don't exist
    for directory in [MODEL_DIR, LOG_DIR, PRED_DIR]:
        os.makedirs(directory, exist_ok=True)

    # Log GPU information
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print(f"GPU available: {gpu}")

        # Set memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print(f"Found {len(gpus)} GPU(s), training will use GPU")
    else:
        print("No GPU found, training will use CPU")

    # Train model
    train_model(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        loss_function=args.loss,
        model_path=args.model_path,
        continue_training=args.continue_training
    )