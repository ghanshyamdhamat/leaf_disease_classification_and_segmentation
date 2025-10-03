# Medical Image Analysis for Disease Classification and Segmentation

This project implements comprehensive machine learning approaches for medical image analysis, focusing on binary classification (healthy vs diseased) and pixel-level segmentation of diseased regions. The project compares traditional machine learning with deep learning approaches and includes statistical analysis of model performance.

## Project Overview

The project consists of four main components:

1. **PCA Analysis** - Dimensionality reduction and variance analysis
2. **SVM Classification** - Traditional machine learning approach with PCA preprocessing
3. **CNN Classification** - Deep learning approach using ResNet18
4. **Image Segmentation** - Pixel-level disease detection using Segformer

## Project Structure

```
├── cnn_classification.ipynb    # CNN-based binary classification
├── SVM_classification.ipynb    # SVM classification with PCA
├── pca.ipynb                  # Principal Component Analysis
├── segmentation.ipynb         # Image segmentation implementation
└── images/                    # Visualization results
    ├── accuracy_cnn_svm.png
    ├── image_at_diff_component.png
    ├── mse.png
    ├── performance_comparison.png
    ├── variance_channel_wise.png
    └── variance_explained.png
```

## Key Features

### 1. CNN Classification ([`cnn_classification.ipynb`](cnn_classification.ipynb))

- **Architecture**: ResNet18 with custom fully connected layer
- **Task**: Binary classification (Healthy vs Diseased)
- **Training Strategy**: 5-fold cross-validation with 3 repeats
- **Data Augmentation**: Random rotation (±10°), horizontal/vertical flips, random cropping
- **Performance**: Achieves ~95% validation accuracy
- **Image Resolution**: 256×256 pixels
- **Batch Size**: 16
- **Learning Rate**: 0.0001
- **Early Stopping**: Implemented with patience mechanism

### 2. SVM Classification ([`SVM_classification.ipynb`](SVM_classification.ipynb))

- **Preprocessing**: PCA dimensionality reduction (95% variance retained)
- **Kernel**: Polynomial kernel SVM
- **Evaluation**: 5-fold cross-validation with ensemble voting
- **Feature Extraction**: Flattened pixel values after PCA transformation
- **Performance**: Achieves ~72% accuracy
- **Statistical Testing**: Paired t-test comparing CNN vs SVM performance

### 3. PCA Analysis ([`pca.ipynb`](pca.ipynb))

- **Implementation**: Custom PCA class using SVD decomposition
- **Analysis**: Channel-wise variance analysis (RGB channels)
- **Visualization**: Reconstruction quality vs number of components
- **Key Finding**: 500 components provide optimal balance (MSE ≈ 0, ~50% dimensionality reduction)
- **Variance Retention**: 95% variance retained for SVM preprocessing

### 4. Image Segmentation ([`segmentation.ipynb`](segmentation.ipynb))

- **Architecture**: Segformer with ResNet34 encoder
- **Loss Function**: Dice Loss for binary segmentation
- **Output Format**: Run-length encoded masks for diseased regions
- **Ensemble Method**: Multi-fold model averaging for robust predictions
- **Evaluation Metrics**: Dice score, F1-score, precision, recall

## Model Performance Comparison

### Classification Results

| Model                      | Mean Accuracy   | Standard Deviation | Best Performance |
| -------------------------- | --------------- | ------------------ | ---------------- |
| **CNN (ResNet18)**   | **95.0%** | ±2.1%             | 97.25%           |
| **SVM (Polynomial)** | 72.0%           | ±3.8%             | 76.8%            |

### Statistical Analysis

- **Paired t-test p-value**: < 0.001 (highly significant)
- **Effect Size (Cohen's d)**: 20.7 (very large effect)
- **95% Confidence Interval**: [0.2291, 0.2417]
- **Conclusion**: CNN significantly outperforms SVM

### Segmentation Performance

- **Dice Score**: Variable depending on disease severity
- **F1-Score**: Calculated for pixel-level accuracy
- **Ensemble Approach**: Averaging predictions from 3 models

## Technical Implementation

### Data Pipeline

1. **Image Loading**: PIL/OpenCV for efficient image processing
2. **Normalization**: ImageNet statistics for pretrained models
3. **Augmentation**: Joint transformations for image-mask pairs
4. **Cross-validation**: Stratified 5-fold splitting

### Model Architectures

- **CNN**: ResNet18 → FC layer → Sigmoid activation
- **SVM**: PCA preprocessing → Polynomial kernel SVM
- **Segmentation**: Segformer encoder-decoder with ResNet34 backbone

### Key Hyperparameters

```python
# CNN Training
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.0001
RESIZE_DIM = (256, 256)
DEVICE = 'cuda' if available else 'cpu'

# SVM Training  
PCA_COMPONENTS = 0.95  # 95% variance retention
KERNEL = 'poly'
RESIZE = (256, 256)

# Segmentation
BATCH_SIZE = 4
EPOCHS = 60
LEARNING_RATE = 0.0001
```

## Data Augmentation Techniques

```python
# Joint transformations for classification
def joint_random_rotation(image, mask, degrees=10)
def joint_random_horizontal_flip(image, mask)
def joint_random_vertical_flip(image, mask)
def joint_random_crop(image, mask, output_size)
```

## Results and Visualizations

The [`images/`](images/) directory contains comprehensive analysis:

- [`accuracy_cnn_svm.png`](images/accuracy_cnn_svm.png) - Performance comparison between models
- [`variance_explained.png`](images/variance_explained.png) - PCA variance analysis
- [`mse.png`](images/mse.png) - Reconstruction error vs components
- [`performance_comparison.png`](images/performance_comparison.png) - Statistical comparison charts

## Key Dependencies

```python
# Deep Learning
torch>=1.9.0
torchvision>=0.10.0
segmentation-models-pytorch

# Traditional ML
scikit-learn>=0.24.0
opencv-python>=4.5.0

# Data Processing
numpy>=1.21.0
pandas>=1.3.0
PIL>=8.3.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Usage Instructions

### 1. Environment Setup

```bash
pip install torch torchvision segmentation-models-pytorch
pip install scikit-learn opencv-python pandas matplotlib seaborn
```

### 2. Running Individual Components

**PCA Analysis:**

```python
# Run pca.ipynb for dimensionality reduction analysis
# Outputs: variance analysis, reconstruction quality
```

**SVM Classification:**

```python
# Run SVM_classification.ipynb for traditional ML approach
# Includes: PCA preprocessing, cross-validation, ensemble voting
```

**CNN Classification:**

```python
# Run cnn_classification.ipynb for deep learning approach
# Features: data augmentation, early stopping, model ensembling
```

**Segmentation:**

```python
# Run segmentation.ipynb for pixel-level disease detection
# Outputs: segmentation masks, run-length encoding
```

### 3. Model Training Pipeline

1. **Data Preparation**: Images resized to 256×256, normalized using ImageNet statistics
2. **Cross-Validation**: 5-fold CV with 3 repeats for robust evaluation
3. **Model Training**: Early stopping based on validation loss
4. **Ensemble Prediction**: Averaging predictions across folds
5. **Statistical Analysis**: Paired t-test for model comparison

## Key Findings

1. **CNN vs SVM**: CNN significantly outperforms SVM (95% vs 72% accuracy)
2. **PCA Effectiveness**: 500 components optimal for dimensionality reduction
3. **Data Augmentation**: Critical for CNN generalization performance
4. **Ensemble Benefits**: Multi-fold averaging improves segmentation robustness
5. **Statistical Significance**: Large effect size (Cohen's d = 20.7) confirms CNN superiority
