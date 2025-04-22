# Brain Tumor Segmentation with U-Net

This repository contains code for automatic segmentation of brain tumors in MRI images using the U-Net architecture.

## Project Overview

This project implements a deep learning solution for segmenting brain tumors in MRI scans. The system uses a U-Net architecture to identify and delineate lower-grade gliomas from multi-channel MRI images, which include pre-contrast, FLAIR, and post-contrast sequences.

## Dataset

The project uses the LGG Segmentation Dataset, which contains brain MR images with manual FLAIR abnormality segmentation masks. The dataset includes:

- Brain MRI images from 23 patients with lower-grade gliomas
- Each image has 3 channels: pre-contrast, FLAIR, and post-contrast
- Binary masks identifying the tumor regions (FLAIR abnormality)

All images are provided in `.tif` format. The dataset should be placed in the `data/` directory.

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd brain_tumor_segmentation
```

2. Create a virtual environment and activate it:
```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Exploration

To explore the dataset and visualize samples:
```bash
jupyter notebook notebooks/data_exploration.ipynb
```

### Training

To train the U-Net model:
```bash
python main.py --mode train --epochs 50 --loss dice_loss
```

Additional options:
- `--batch_size`: Set the batch size (default: 8)
- `--lr`: Set the learning rate (default: 1e-4)
- `--model_path`: Path to a pre-trained model to continue training
- `--continue_training`: Continue training from a checkpoint
- `--evaluate_after_training`: Evaluate model after training

### Evaluation

To evaluate a trained model:
```bash
python main.py --mode evaluate --model_path outputs/models/unet_brain_tumor_best.h5
```

### Prediction

To make predictions on a single image:
```bash
python main.py --mode predict --image_path data/patient_folder/image.tif
```

### Visualization

To visualize samples from the dataset:
```bash
python main.py --mode visualize --num_samples 5
```

## Model Architecture

The project implements the U-Net architecture, consisting of:

1. **Encoder Path**: A contracting path with convolutional blocks and max pooling operations that reduce spatial dimensions while increasing feature depth.

2. **Bridge**: The bottleneck connecting the encoder and decoder.

3. **Decoder Path**: An expansive path with upsampling operations and concatenations with features from the encoder path via skip connections.

4. **Output Layer**: A 1×1 convolution with sigmoid activation to produce a binary segmentation mask.

## Evaluation Metrics

The model is evaluated using several metrics:

- **Dice Coefficient**: Measures the overlap between the predicted segmentation and ground truth
- **IoU (Intersection over Union)**: Measures the area of overlap divided by the area of union
- **Accuracy**: Percentage of correctly classified pixels
- **Precision & Recall**: Measures of model's exactness and completeness

## Requirements

- Python 3.11
- TensorFlow 2.8+
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-image
- scikit-learn
- PIL (Pillow)
- tqdm

See `requirements.txt` for the full list of dependencies.

## License

[Include license information here]

## Acknowledgments

- The LGG Segmentation Dataset was obtained from Kaggle.com
- U-Net architecture was originally proposed by Ronneberger et al. in the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation"# Brain_tumor_segmentation
