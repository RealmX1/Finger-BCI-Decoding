# Finger-BCI-Decoding

EEG-based Brain-Computer Interface for Real-time Individual Finger-Level Robotic Hand Control.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.16+](https://img.shields.io/badge/tensorflow-2.16%2B-orange.svg)](https://www.tensorflow.org/)
[![Keras 3](https://img.shields.io/badge/keras-3.x-red.svg)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This repository contains the implementation for the paper:

> **EEG-based Brain-Computer Interface Enables Real-time Robotic Hand Control at Individual Finger Level**
>
> Ding, Y., Udompanyawit, C., Zhang, Y., & He, B. (2025). *Nature Communications*, 16(1), 5401.
>
> DOI: [10.1038/s41467-025-61064-x](https://doi.org/10.1038/s41467-025-61064-x)

The system decodes motor imagery (MI) and motor execution (ME) EEG signals to control individual fingers of a robotic hand in real-time using deep learning (EEGNet architecture).

## Features

- **EEGNet-8,2 Architecture**: Compact CNN optimized for EEG-based BCIs
- **5-Fold Stratified Cross-Validation**: Robust model evaluation
- **Transfer Learning**: Fine-tuning with layer freezing for improved performance
- **Online Smoothing**: Temporal probability smoothing for stable control
- **Majority Voting**: Trial-level prediction from sliding window segments
- **Complete Evaluation Pipeline**: Precision, Recall, and Confusion Matrix metrics

## Installation

### Prerequisites

- Python 3.10 or 3.11
- NVIDIA GPU (recommended for training)
- Conda (for GPU setup) or [uv](https://github.com/astral-sh/uv) (for CPU-only)

### Option 1: GPU Training (Recommended)

For NVIDIA RTX 50 series (Blackwell) or newer GPUs with compute capability 12.0+:

```bash
# Create conda environment
conda create --name tf_gpu python=3.11.4 pip -y
conda activate tf_gpu

# Install TensorFlow Nightly (has compute capability 12.0 support)
pip install tf-nightly

# Install CUDA libraries
pip install nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 \
    nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusparse-cu12 \
    nvidia-cusolver-cu12 nvidia-cuda-nvrtc-cu12

# Configure library path
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib
conda deactivate && conda activate tf_gpu

# Install other dependencies
pip install scipy scikit-learn matplotlib

# Verify GPU
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

**GPU Setup Guides**:
- RTX 50 series (Blackwell): [docs/RTX50_GPU_SETUP.md](docs/RTX50_GPU_SETUP.md)
- RTX 40/30/20 series + WSL2: [docs/WSL2_GPU_SETUP.md](docs/WSL2_GPU_SETUP.md)

### Option 2: CPU-Only (Quick Start)

```bash
# Clone the repository
git clone https://github.com/your-username/Finger-BCI-Decoding.git
cd Finger-BCI-Decoding

# Create virtual environment with uv
uv venv --python 3.10
uv sync

# Verify installation
uv run python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

## Project Structure

```
Finger-BCI-Decoding/
├── Functions.py                 # Core data loading and preprocessing
├── EEGModels_tf.py             # EEGNet model architecture
├── main_model_training.py      # Training entry script
├── main_online_processing.py   # Real-time BCI processing (BCPy2000)
│
├── training/
│   └── cross_validation.py     # 5-fold stratified cross-validation
│
├── evaluation/
│   └── test_evaluation.py      # Majority voting & metrics calculation
│
├── online/
│   └── online_smoothing.py     # Temporal probability smoothing
│
├── scripts/
│   └── run_experiment.py       # Complete experiment pipeline
│
├── results/                    # Experiment outputs
├── models/                     # Saved model files (.keras)
└── docs/
    ├── RTX50_GPU_SETUP.md     # RTX 50 series GPU setup
    └── WSL2_GPU_SETUP.md      # WSL2 GPU setup (RTX 40/30/20)
```

## Usage

### Running Complete Experiment

```bash
# Activate GPU environment
conda activate tf_gpu

# Binary classification (Thumb vs Pinky) for Subject 1
python scripts/run_experiment.py \
    --subj 1 \
    --task MI \
    --nclass 2 \
    --data-folder ./data
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--subj` | Subject ID (1-21) | Required |
| `--task` | Task type: `MI` (Motor Imagery) or `ME` (Motor Execution) | Required |
| `--nclass` | Number of classes (2 or 3) | 2 |
| `--sessions` | Session numbers to run | 1 2 |
| `--data-folder` | Path to data directory | Required |
| `--save-folder` | Output directory | ./results |
| `--no-cv` | Skip cross-validation | False |

### Training Individual Models

```bash
# Activate environment
conda activate tf_gpu

# Train Base model (pre-training)
python main_model_training.py 1 1 2 MI Orig --data-folder ./data --save-folder ./models

# Train Finetune model (transfer learning)
python main_model_training.py 1 1 2 MI Finetune --data-folder ./data --save-folder ./models
```

Arguments: `subj_id session_num nclass task modeltype`

## Data Format

Expected directory structure:

```
data/
└── S01/
    ├── OfflineImagery/              # Offline training data
    │   └── *.mat                    # MATLAB files with EEG data
    ├── OnlineImagery_Sess01_2class_Base/      # Session 1 Base test
    ├── OnlineImagery_Sess01_2class_Finetune/  # Session 1 Finetune test
    ├── OnlineImagery_Sess02_2class_Base/      # Session 2 Base test
    └── OnlineImagery_Sess02_2class_Finetune/  # Session 2 Finetune test
```

Each `.mat` file contains:
- `eeg.data`: EEG signals (channels × samples)
- `eeg.fsample`: Sampling rate (1024 Hz)
- `event`: Trial events with type, sample, and value

## Model Architecture

**EEGNet-8,2** configuration:

| Parameter | Value |
|-----------|-------|
| Temporal filters (F1) | 8 |
| Depth multiplier (D) | 2 |
| Pointwise filters (F2) | 16 |
| Kernel length | 32 |
| Dropout (Base) | 0.5 |
| Dropout (Finetune) | 0.65 |

## Training Pipeline

### Base Model Training
1. Load offline data + previous session data (cumulative)
2. 5-fold stratified cross-validation
3. Dynamic class weight balancing
4. Early stopping (patience=10)
5. Learning rate reduction on plateau

### Fine-tuning
1. Load current session's Base test data
2. Initialize from pre-trained Base model
3. Freeze first 4 layers
4. Train with reduced learning rate (1e-4)
5. Higher dropout (0.65) for regularization

## Evaluation Metrics

- **Segment-level Accuracy**: Per-window predictions
- **Majority Voting Accuracy**: Trial-level predictions via voting
- **Per-class Precision/Recall**: Thumb and Pinky performance
- **Confusion Matrix**: Detailed classification breakdown

## Expected Results

Binary classification (Thumb vs Pinky) MI task:

| Model | Expected Accuracy |
|-------|------------------|
| Session 1 Base | ~70-75% |
| Session 1 Finetune | ~75-80% |
| Session 2 Finetune | ~80.56% (paper) |

## Online Processing

The `main_online_processing.py` script integrates with BCPy2000 for real-time BCI control. Key features:

- 128-sample block processing (125 ms @ 1024 Hz)
- Online bandpass filtering (4-40 Hz)
- Probability smoothing: `P'_t = α * h_{t-1} + P_t`
- Real-time robotic hand command generation

## Dependencies

### GPU Training (tf_gpu environment)
- TensorFlow Nightly 2.21+ (for RTX 50 series)
- Keras 3.x
- NumPy 2.x
- SciPy 1.16+
- scikit-learn 1.8+

### CPU Training (uv environment)
- TensorFlow 2.16+
- Keras 3.x
- NumPy 1.x or 2.x
- SciPy 1.10+
- scikit-learn 1.2+

**Note**: This codebase is compatible with Keras 3. Model files use the native `.keras` format.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ding2025eeg,
  title={EEG-based brain-computer interface enables real-time robotic hand control at individual finger level},
  author={Ding, Yidan and Udompanyawit, Chatchai and Zhang, Yixuan and He, Bin},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={5401},
  year={2025},
  publisher={Nature Publishing Group},
  doi={10.1038/s41467-025-61064-x}
}
```

## Acknowledgments

- EEGNet implementation adapted from [ARL EEGModels](https://github.com/vlawhern/arl-eegmodels)
- This work was supported by NIH grants NS124564, NS131069, NS127849, and NS096761

## References

1. Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15, 056013.

## Changelog

### v2.0.0 (2026-01)
- **Keras 3 Compatibility**: Updated codebase for TensorFlow 2.16+ and Keras 3.x
  - Replaced `tf.keras.optimizers.legacy.Adam` with `tf.keras.optimizers.Adam`
  - Updated model save format from `.h5` to native `.keras` format
- **RTX 50 Series Support**: Added setup guide for Blackwell architecture GPUs
- **Documentation**: Reorganized installation guides for different GPU generations

### v1.0.0 (2025)
- Initial release accompanying Nature Communications publication
- TensorFlow 2.10 with Keras 2.x

## License

MIT License - see [LICENSE](LICENSE) for details.
