"""
Main script for brain tumor segmentation.
"""
import os
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import time

from src.config import *
from src.data_loader import get_data_generators
from src.model import build_unet, compile_model, get_model_callbacks, dice_coefficient, dice_loss
from src.train import train_model
from src.evaluate import evaluate_model
from src.data_loader import get_image_mask_paths
from src.utils import predict_and_visualize, visualize_dataset_samples


def main(args):
    """
    Main function for brain tumor segmentation.

    Args:
        args: Command line arguments
    """
    # Set GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU found, using CPU")

    # Create output directories if they don't exist
    for directory in [OUTPUT_DIR, MODEL_DIR, LOG_DIR, PRED_DIR]:
        os.makedirs(directory, exist_ok=True)

    # Get all image and mask pairs
    image_mask_pairs = get_image_mask_paths()
    print(f"Found {len(image_mask_pairs)} image-mask pairs")

    if args.mode == 'train':
        print("\n===== TRAINING MODE =====")

        # Train model
        model, history = train_model(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            loss_function=args.loss,
            model_path=args.model_path,
            continue_training=args.continue_training
        )

        # Evaluate model if specified
        if args.evaluate_after_training:
            print("\n===== EVALUATING MODEL =====")
            evaluate_model(
                model_path=os.path.join(MODEL_DIR, 'unet_brain_tumor_best.h5'),
                batch_size=args.batch_size,
                save_predictions=True
            )

    elif args.mode == 'evaluate':
        print("\n===== EVALUATION MODE =====")

        # Evaluate model
        evaluate_model(
            model_path=args.model_path,
            batch_size=args.batch_size,
            save_predictions=True
        )

    elif args.mode == 'predict':
        print("\n===== PREDICTION MODE =====")

        if not os.path.exists(args.model_path):
            print(f"Error: Model file not found at {args.model_path}")
            return

        if not args.image_path:
            print("Error: Image path not specified. Use --image_path to specify an image for prediction")
            return

        # Load model
        print(f"Loading model from {args.model_path}...")
        custom_objects = {
            'dice_coefficient': dice_coefficient,
            'dice_loss': dice_loss
        }
        model = load_model(args.model_path, custom_objects=custom_objects)
        print("Model loaded successfully")

        # Make prediction
        print(f"Making prediction on {args.image_path}...")
        mask_path = args.image_path.replace('.tif', '_mask.tif') if os.path.exists(args.image_path.replace('.tif', '_mask.tif')) else None
        save_path = os.path.join(PRED_DIR, f"prediction_{os.path.basename(args.image_path).replace('.tif', '.png')}")

        prediction = predict_and_visualize(model, args.image_path, mask_path, save_path)
        print(f"Prediction saved to {save_path}")

    elif args.mode == 'visualize':
        print("\n===== VISUALIZATION MODE =====")

        # Visualize dataset samples
        print(f"Visualizing {args.num_samples} random samples from the dataset...")
        visualize_dataset_samples(
            image_mask_pairs,
            num_samples=args.num_samples,
            save_dir=os.path.join(OUTPUT_DIR, 'visualizations')
        )
        print("Visualization completed")

    else:
        print(f"Error: Unknown mode {args.mode}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Brain Tumor Segmentation')

    # Common arguments
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'predict', 'visualize'], default='train',
                        help='Mode: train, evaluate, predict, or visualize')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--loss', type=str, default=LOSS_FUNCTION, choices=['binary_crossentropy', 'dice_loss'],
                        help='Loss function')
    parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_DIR, 'unet_brain_tumor_best.h5'),
                        help='Path to a pre-trained model')
    parser.add_argument('--continue_training', action='store_true', help='Continue training a pre-trained model')
    parser.add_argument('--evaluate_after_training', action='store_true', help='Evaluate model after training')

    # Prediction arguments
    parser.add_argument('--image_path', type=str, default=None, help='Path to the image for prediction (used in prediction mode)')

    # Visualization arguments
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize (used in visualization mode)')

    args = parser.parse_args()

    # Start timer
    start_time = time.time()

    # Run main function
    main(args)

    # End timer
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds ({(end_time - start_time) / 60:.2f} minutes)")