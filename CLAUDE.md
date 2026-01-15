# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a facial beauty scoring system (颜值打分系统) that combines traditional geometric analysis with deep learning models. The project has two main scoring approaches:

1. **Geometric Scoring**: Uses traditional aesthetic standards (Three-Regions, Five-Eyes, symmetry) - works immediately without training
2. **Deep Learning Scoring**: Uses neural networks (AlexNet, ResNet-18, ResNeXt-50) trained on SCUT-FBP5500 dataset

## Core Architecture

### Module Structure

- **`analyzer/`**: Face detection and landmark extraction using MediaPipe (468 landmarks)
- **`scorer/`**: Geometric feature scoring algorithms
- **`models/`**: Deep learning model definitions (uses `get_model()` function to retrieve models)
- **`dataset/`**: SCUT-FBP5500 dataset loader
- **`utils/`**: Geometry calculations, metrics, and image processing utilities
- **`config.py`**: Centralized training configuration (auto-detects GPU on both Windows and Linux)

### Entry Points

- **`main.py`**: Main CLI entry point supporting both geometric and DL scoring
- **`train.py`**: Training script for deep learning models
- **`evaluate.py`**: Model evaluation using Pearson Correlation, MAE, RMSE
- **`predict.py`**: Prediction using trained models

## Common Commands

### Installation
```bash
pip install -r requirements.txt
python -c "import torch, cv2, mediapipe; print('OK')"
```

### Geometric Scoring (No Training Required)
```bash
python main.py <image_path>
python main.py <image_path> --output report.txt --verbose
```

### Deep Learning Pipeline

**Dataset preparation:**
```bash
python prepare_data.py --zip dataset/SCUT-FBP5500_v2.zip --extract-to dataset
python prepare_data.py --check-only --extract-to dataset/SCUT-FBP5500_v2
```

**Training:**
```bash
# Recommended: ResNet-18 (best balance)
python train.py --model resnet18 --dataset dataset/SCUT-FBP5500_v2 --gpu 0

# Faster: AlexNet
python train.py --model alexnet --dataset dataset/SCUT-FBP5500_v2 --gpu 0

# Highest accuracy: ResNeXt-50
python train.py --model resnext50 --dataset dataset/SCUT-FBP5500_v2 --gpu 0
```

**Evaluation:**
```bash
python evaluate.py --checkpoint outputs/checkpoints/resnet18_best.pth --dataset dataset/SCUT-FBP5500_v2
```

**Prediction:**
```bash
python predict.py <image_path> --checkpoint outputs/checkpoints/resnet18_best.pth --gpu
```

**TensorBoard:**
```bash
tensorboard --logdir outputs/logs
```

## Key Architecture Details

### Geometric Scoring Algorithm (35/30/35 weights)
- **Three-Regions (三庭)**: Upper/Middle/Lower face proportions (ideal 1:1:1)
- **Five-Eyes (五眼)**: Face width equals 5 eye widths
- **Symmetry**: Left-right facial symmetry analysis

### Deep Learning Training Configuration (SCUT-FBP5500 paper-based)
- **Image sizes**: 256x256 input, crop to 227x227 (AlexNet) or 224x224 (ResNet/ResNeXt)
- **Pretraining**: ImageNet weights
- **Optimizer**: SGD (momentum=0.9, weight_decay=5e-4 for AlexNet, 1e-4 for ResNet)
- **Learning rate**: 0.001, decays by 0.1 every 5000 iterations
- **Loss**: MSE (L2-norm distance)
- **Metrics**: Pearson Correlation (PC), MAE, RMSE

### Dataset Handling

The `Config` class automatically finds split files with this priority:
1. `train_test_files/split_of_60%training and 40%testing/train.txt` (actual folder name with spaces)
2. `train_test_files/split_of_60%_training_and_40%_testing/train.txt` (underscore version)
3. `train_test_files/5_folders_cross_validations_files/cross_validation_1/train_1.txt`

Images are expected in `dataset/SCUT-FBP5500_v2/Images/` directory. The dataset loader handles both cases automatically.

### Model Retrieval

Always use `get_model(model_name, pretrained=True)` from the `models` module:
```python
from models import get_model
model = get_model('resnet18', pretrained=True)
```

### GPU Detection

The codebase uses this pattern for GPU detection (Windows/Linux compatible):
```python
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

## Expected Performance (SCUT-FBP5500 paper)

| Model    | PC     | MAE    | RMSE   |
|----------|--------|--------|--------|
| AlexNet  | 0.8634 | 0.2651 | 0.3481 |
| ResNet-18| 0.89   | 0.2419 | 0.3166 |
| ResNeXt-50| 0.8997| 0.2291 | 0.3017 |

## Current Status

- Phase 1 (Face analysis): Complete
- Phase 2 (Geometric scoring): Complete
- Phase 2 (Deep learning): Complete
- Phase 3 (AIGC beautification with Stable Diffusion): Planned
