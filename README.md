# Finger-BCI-Decoding

EEG-based Brain-Computer Interface for Real-time Individual Finger-Level Robotic Hand Control.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10](https://img.shields.io/badge/tensorflow-2.10-orange.svg)](https://www.tensorflow.org/)
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

- Python 3.10
- [uv](https://github.com/astral-sh/uv) package manager
- NVIDIA GPU (optional, for accelerated training via WSL2)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/Finger-BCI-Decoding.git
cd Finger-BCI-Decoding

# Create virtual environment and install dependencies
uv venv --python 3.10
uv sync

# Verify installation
uv run python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

### GPU Acceleration (WSL2)

For GPU-accelerated training on Windows, see [docs/WSL2_GPU_SETUP.md](docs/WSL2_GPU_SETUP.md).

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
└── docs/
    └── WSL2_GPU_SETUP.md      # GPU setup guide
```

## Usage

### Running Complete Experiment

```bash
# Binary classification (Thumb vs Pinky) for Subject 1
uv run python scripts/run_experiment.py \
    --subj 1 \
    --task MI \
    --nclass 2 \
    --data-folder /path/to/data
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
# Train Base model (pre-training)
uv run python main_model_training.py 1 1 2 MI Orig

# Train Finetune model (transfer learning)
uv run python main_model_training.py 1 1 2 MI Finetune
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
4. Early stopping (patience=80)
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

- TensorFlow 2.10.1
- NumPy 1.23.x
- SciPy 1.10-1.11
- scikit-learn 1.2-1.3
- Matplotlib 3.7+

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

## License

MIT License - see [LICENSE](LICENSE) for details.
