#!/usr/bin/env python3
"""
Inference script for applying a trained Sparse Autoencoder to neural data

Usage:
    python inference.py --data_path /path/to/neural_data.npy --model_path /path/to/model.pt
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from models.sparse_autoencoder import SparseAutoencoder
from data.dataloader import NeuralDataset, get_neural_dataloader
from utils.visualization import plot_reconstructions, plot_dictionary_elements, plot_encoding_statistics


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Apply a trained sparse autoencoder to neural data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to neural data file (.npy or .npz)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./inference_results', help='Directory to save inference results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # Create dataloader
    dataloader, dataset = get_neural_dataloader(
        data=neural_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Load model
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Determine the embedding dimension from the saved model
    if 'model_state_dict' in checkpoint:
        # Get encoder weight shape to determine embedding_dim
        encoder_weight = checkpoint['model_state_dict']['encoder.weight']
        embedding_dim = encoder_weight.shape[0]
    else:
        # Fallback if we can't determine from the checkpoint
        print("Could not determine embedding dimension from checkpoint, using 1024")
        embedding_dim = 1024
    
    print(f"Embedding dimension: {embedding_dim}")
    
    # Create the sparse autoencoder model
    model = SparseAutoencoder(
        input_dim=input_dim,
        embedding_dim=embedding_dim
    )
    
    # Load the state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Generate encodings for all data
    print("Generating encodings for all samples...")
    all_encodings = []
    all_reconstructions = []
    all_inputs = []
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            reconstructed, encoded, _ = model(batch)
            
            # Calculate reconstruction error for each sample
            errors = torch.mean((batch - reconstructed) ** 2, dim=1)
            
            all_encodings.append(encoded.cpu().numpy())
            all_reconstructions.append(reconstructed.cpu().numpy())
            all_inputs.append(batch.cpu().numpy())
            reconstruction_errors.append(errors.cpu().numpy())
    
    all_encodings = np.vstack(all_encodings)
    all_reconstructions = np.vstack(all_reconstructions)
    all_inputs = np.vstack(all_inputs)
    reconstruction_errors = np.concatenate(reconstruction_errors)
    
    # Save the encodings
    np.save(os.path.join(args.output_dir, 'encodings.npy'), all_encodings)
    np.save(os.path.join(args.output_dir, 'reconstruction_errors.npy'), reconstruction_errors)
    
    print(f"Generated encodings shape: {all_encodings.shape}")
    print(f"Average reconstruction error: {np.mean(reconstruction_errors):.6f}")
    
    # Plot and save the results
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot sample reconstructions
    print("Plotting reconstructions...")
    recon_fig = plot_reconstructions(model, dataloader, device, n_samples=args.num_samples)
    recon_fig.savefig(os.path.join(plots_dir, 'reconstructions.png'))
    
    # Plot dictionary elements
    print("Plotting dictionary elements...")
    dict_fig = plot_dictionary_elements(model, n_elements=20)
    dict_fig.savefig(os.path.join(plots_dir, 'dictionary_elements.png'))
    
    # Plot encoding statistics
    print("Plotting encoding statistics...")
    stats_fig, _ = plot_encoding_statistics(model, dataloader, device)
    stats_fig.savefig(os.path.join(plots_dir, 'encoding_statistics.png'))
    
    # Plot reconstruction error histogram
    plt.figure(figsize=(10, 6))
    plt.hist(reconstruction_errors, bins=50)
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Mean Squared Error')
    plt.ylabel('Count')
    plt.savefig(os.path.join(plots_dir, 'reconstruction_error_hist.png'))
    
    # Plot the worst reconstructed samples
    worst_indices = np.argsort(reconstruction_errors)[-args.num_samples:]
    
    fig, axes = plt.subplots(args.num_samples, 2, figsize=(12, 3 * args.num_samples))
    
    for i, idx in enumerate(worst_indices):
        sample = all_inputs[idx]
        recon = all_reconstructions[idx]
        
        # Try to reshape to 2D if square
        n_dims = int(np.sqrt(sample.shape[0]))
        try:
            sample_2d = sample.reshape(n_dims, -1)
            recon_2d = recon.reshape(n_dims, -1)
            
            axes[i, 0].imshow(sample_2d, cmap='viridis', aspect='auto')
            axes[i, 1].imshow(recon_2d, cmap='viridis', aspect='auto')
        except:
            axes[i, 0].plot(sample)
            axes[i, 1].plot(recon)
            
        axes[i, 0].set_title(f"Original (Error: {reconstruction_errors[idx]:.4f})")
        axes[i, 1].set_title(f"Reconstruction")
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'worst_reconstructions.png'))
    
    print(f"Inference complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()