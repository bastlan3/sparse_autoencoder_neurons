# Sparse Autoencoder for Neural Data

A PyTorch implementation of a sparse autoencoder designed for neural data analysis with dictionary learning.

## Project Overview

This project implements a sparse autoencoder with linear encoding and decoding layers that can learn meaningful representations from neural data. The model is specifically designed to handle neural recordings with a shape of [neurons, time, presentations], where:

- `neurons`: Number of neurons recorded (e.g., 64)
- `time`: Number of time steps in each recording
- `presentations`: Number of different image presentations or trials

The sparse autoencoder uses L1 regularization to enforce sparsity, implementing dictionary learning to find meaningful latent representations of neural activity.

## Features

- Linear encoding and decoding layers
- L1 sparsity constraint for dictionary learning
- Overcomplete representations (embedding dimension larger than input)
- Comprehensive visualization tools
- Modular design for easy extension

## Project Structure

```
sparse_autoencoder/
├── train.py                    # Main training script
├── inference.py                # Script for using trained model
├── configs/
│   └── config.py               # Configuration and hyperparameters
├── data/
│   └── dataloader.py           # Data loading utilities
├── models/
│   └── sparse_autoencoder.py   # Model architecture
└── utils/
    ├── training.py             # Training loop and utilities
    └── visualization.py        # Visualization functions
```

## Usage

### Training

```bash
python train.py --data_path /path/to/neural_data.npy --output_dir ./results
```

Optional arguments:
- `--embedding_dim`: Dimension of the embedding layer (default: 1024)
- `--sparsity_weight`: Weight for L1 sparsity constraint (default: 0.001)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 32)

### Inference

```bash
python inference.py --data_path /path/to/neural_data.npy --model_path ./results/checkpoints/best_model.pt
```

Optional arguments:
- `--output_dir`: Directory to save inference results (default: ./inference_results)
- `--batch_size`: Batch size for inference (default: 32)
- `--num_samples`: Number of samples to visualize (default: 5)

## Requirements

- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- tqdm
- seaborn