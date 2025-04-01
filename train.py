#!/usr/bin/env python3
"""
Main script for training the Sparse Autoencoder on neural data

Usage:
    python train.py --data_path /path/to/neural_data.npy

The neural data should have shape [neurons, time, presentations]
"""

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import random_split
import matplotlib.pyplot as plt

# Import project modules
from models.sparse_autoencoder import SparseAutoencoder
from data.dataloader import NeuralDataset, get_neural_dataloader
from utils.training import Trainer
from utils.visualization import (
    plot_training_history,
    plot_reconstructions,
    plot_dictionary_elements,
    plot_encoding_statistics
)
from configs.config import data_config, model_config, training_config, viz_config


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a sparse autoencoder on neural data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to neural data file (.npy or .npz)')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save outputs')
    parser.add_argument('--embedding_dim', type=int, default=model_config['embedding_dim'],
                      help='Dimension of embedding layer (default: from config)')
    parser.add_argument('--sparsity_weight', type=float, default=model_config['sparsity_weight'],
                      help='Weight for L1 sparsity constraint (default: from config)')
    parser.add_argument('--epochs', type=int, default=training_config['epochs'],
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=data_config['batch_size'],
                      help='Batch size for training')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    training_config['save_dir'] = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(training_config['save_dir'], exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load neural data
    print(f"Loading neural data from {args.data_path}")
    try:
        if args.data_path.endswith('.npy'):
            neural_data = np.load(args.data_path)
        elif args.data_path.endswith('.npz'):
            loaded = np.load(args.data_path)
            neural_data = loaded[list(loaded.keys())[0]]
        else:
            raise ValueError(f"Unsupported file format: {args.data_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please provide a valid .npy or .npz file with neural data of shape [neurons, time, presentations]")
        return

    # Print data shape information
    print(f"Neural data shape: {neural_data.shape}")
    n_neurons, time_steps, n_presentations = neural_data.shape
    input_dim = n_neurons * time_steps
    print(f"Input dimension: {input_dim} (neurons={n_neurons}, time_steps={time_steps})")
    
    # Create full dataset
    dataset = NeuralDataset(data=neural_data)
    
    # Split dataset into training and validation sets
    train_size = int(data_config['train_val_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=data_config['shuffle'],
        num_workers=data_config['num_workers']
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=data_config['num_workers']
    )
    
    # Create the sparse autoencoder model
    print(f"Creating sparse autoencoder with embedding dim: {args.embedding_dim}, sparsity weight: {args.sparsity_weight}")
    model = SparseAutoencoder(
        input_dim=input_dim,
        embedding_dim=args.embedding_dim,
        sparsity_weight=args.sparsity_weight
    )
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        save_dir=training_config['save_dir'],
        save_freq=training_config['save_freq'],
        patience=training_config['patience']
    )
    
    # Train the model
    print(f"Starting training for {args.epochs} epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs
    )
    
    # Save results
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot training history
    history_fig = plot_training_history(history)
    history_fig.savefig(os.path.join(plots_dir, 'training_history.png'))
    
    # Plot sample reconstructions
    recon_fig = plot_reconstructions(model, val_loader, device, n_samples=viz_config['n_reconstruction_samples'])
    recon_fig.savefig(os.path.join(plots_dir, 'reconstructions.png'))
    
    # Plot dictionary elements
    dict_fig = plot_dictionary_elements(model, n_elements=viz_config['n_dictionary_elements'])
    dict_fig.savefig(os.path.join(plots_dir, 'dictionary_elements.png'))
    
    # Plot encoding statistics
    stats_fig, encodings = plot_encoding_statistics(model, val_loader, device)
    stats_fig.savefig(os.path.join(plots_dir, 'encoding_statistics.png'))
    
    print(f"Training complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()