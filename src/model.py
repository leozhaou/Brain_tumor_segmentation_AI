"""
U-Net model architecture for brain tumor segmentation.
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization,
    Conv2DTranspose, concatenate, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
import tensorflow.keras.backend as K
import os
from src.config import *


def dice_coefficient(y_true, y_pred, smooth=1.0):
    """
    Dice coefficient for binary segmentation.

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice coefficient
    """
    # Cast both tensors to the same data type (float32) to prevent type mismatch
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred, smooth=1.0):
    """
    Dice loss for binary segmentation.

    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice loss
    """
    return 1 - dice_coefficient(y_true, y_pred, smooth)


def build_unet(input_shape=(None, None, CHANNELS), filters_base=FILTERS_BASE, dropout_rate=DROPOUT_RATE):
    # Input layer
    inputs = Input(input_shape)

    # Contracting path (encoder)
    # Block 1
    conv1 = Conv2D(filters_base, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(filters_base, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Block 2
    conv2 = Conv2D(filters_base*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(filters_base*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Block 3
    conv3 = Conv2D(filters_base*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(filters_base*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Block 4
    conv4 = Conv2D(filters_base*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(filters_base*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(dropout_rate)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bridge
    conv5 = Conv2D(filters_base*16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(filters_base*16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(dropout_rate)(conv5)

    # Expansive path (decoder)
    # Block 6
    up6 = Conv2DTranspose(filters_base*8, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(drop5)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(filters_base*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(filters_base*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    # Block 7
    up7 = Conv2DTranspose(filters_base*4, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(filters_base*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(filters_base*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    # Block 8
    up8 = Conv2DTranspose(filters_base*2, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(filters_base*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(filters_base*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    # Block 9
    up9 = Conv2DTranspose(filters_base, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(filters_base, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(filters_base, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)

    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    # Model compilation
    model = Model(inputs=inputs, outputs=outputs)

    return model


def get_model_callbacks(model_name='unet_brain_tumor'):
    """
    Get callbacks for model training.

    Args:
        model_name: Name of the model for saving

    Returns:
        List of callbacks
    """
    # Model checkpoint callback
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, f'{model_name}_best.h5'),
        monitor='val_dice_coefficient',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_dice_coefficient',
        mode='max',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )

    # Reduce learning rate on plateau callback
    reduce_lr = ReduceLROnPlateau(
        monitor='val_dice_coefficient',
        mode='max',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    # TensorBoard callback
    tensorboard = TensorBoard(
        log_dir=os.path.join(LOG_DIR, model_name),
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch',
        profile_batch=0
    )

    # CSV logger callback
    csv_logger = CSVLogger(
        os.path.join(LOG_DIR, f'{model_name}_training.log'),
        separator=',',
        append=False
    )

    return [model_checkpoint, early_stopping, reduce_lr, tensorboard, csv_logger]


def compile_model(model, learning_rate=LEARNING_RATE, loss_function=LOSS_FUNCTION):
    """
    Compile the model with appropriate loss function and metrics.

    Args:
        model: U-Net model
        learning_rate: Learning rate for the optimizer
        loss_function: Loss function name ('binary_crossentropy' or 'dice_loss')

    Returns:
        Compiled model
    """
    # Define optimizer
    optimizer = Adam(learning_rate=learning_rate)

    # Define loss function
    if loss_function == 'binary_crossentropy':
        loss = 'binary_crossentropy'
    elif loss_function == 'dice_loss':
        loss = dice_loss
    else:
        raise ValueError(f"Unsupported loss function: {loss_function}")

    # Define metrics
    metrics = ['accuracy', dice_coefficient]

    # Compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


if __name__ == "__main__":
    # Test the model
    print("Testing U-Net model...")

    # Build model
    input_shape = (*IMAGE_SIZE, CHANNELS)
    model = build_unet(input_shape=input_shape)

    # Print model summary
    model.summary()

    # Compile model
    model = compile_model(model)

    # Get callbacks
    callbacks = get_model_callbacks()

    print("Model test completed successfully")