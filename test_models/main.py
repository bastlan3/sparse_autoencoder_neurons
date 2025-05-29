#!/usr/bin/env python3
"""
Main training script for sparse dictionary learning on ResNet50 features.
Recreates findings from "Towards Monosemanticity" paper.

This script orchestrates the training process, which involves:
1.  **ResNet Feature Extraction**: Utilizes a pre-trained ResNet model (e.g., ResNet50)
    to extract feature vectors from input images. These features serve as the
    input to the sparse dictionary.
2.  **Sparse Dictionary Learning**: Trains a sparse dictionary to reconstruct the
    ResNet features using a sparse set of dictionary elements (atoms). The goal is
    to learn a meaningful and interpretable representation of the features.
3.  **Data Handling**: Loads datasets like ImageNet and potentially others (e.g., Many00)
    for training and validation.
4.  **Experiment Modes**:
    -   **Full Experiment**: Uses all features extracted by the ResNet layer.
    -   **Subset Experiment**: Focuses on a small subset of neurons (e.g., 64) from
        the ResNet features, allowing for more targeted analysis.
5.  **Training Loop**: Implements the main training logic, including:
    -   Forward and backward passes.
    -   Calculation of reconstruction and sparsity losses.
    -   Optimization using Adam and learning rate scheduling.
    -   Periodic validation and model checkpointing.
    -   Logging to Weights & Biases (wandb) for experiment tracking.
    -   Visualization of dictionary features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import json
import wandb
from tqdm import tqdm

from models.sparse_dictionary import SparseDictionary
from models.resnet_extractor import ResNetFeatureExtractor
from data.imagenet_loader import ImageNetDataLoader
from data.many00_loader import Many00DataLoader
from utils.training_utils import compute_sparsity_loss, compute_reconstruction_loss
from utils.visualization_utils import visualize_dictionary_features, run_post_training_analysis


def main():
    # Change working directory to where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    original_cwd = os.getcwd()
    os.chdir(script_dir)
    print(f"Changed working directory from {original_cwd} to {script_dir}")
    
    parser = argparse.ArgumentParser(description='Sparse Dictionary Learning on ResNet50')
    parser.add_argument('--experiment_name', type=str, default='sparse_dict_resnet50',
                       help='Name for the experiment')
    parser.add_argument('--use_subset', action='store_true',
                       help='Use only 64 neurons subset experiment')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    
    # Model arguments
    parser.add_argument('--resnet_model', type=str, default='resnet50',
                       help='ResNet model name (resnet18, resnet50, etc.)')
    parser.add_argument('--target_layer', type=str, default='layer4',
                       help='Target layer for feature extraction')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained ResNet model')
    
    # Dictionary arguments
    parser.add_argument('--dict_size', type=int, default=4000,
                       help='Dictionary size')
    parser.add_argument('--sparsity_coef', type=float, default=0.1,
                       help='Sparsity coefficient')
    
    # Data arguments
    parser.add_argument('--imagenet_path', type=str, default='/tmp/imagenet',
                       help='Path to ImageNet dataset')
    parser.add_argument('--many00_path', type=str, default='./../../../data/manyOO',
                       help='Path to Many00 dataset')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--use_imagenet', action='store_true',
                       help='Use ImageNet dataset instead of Many00')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=254,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval (batches)')
    parser.add_argument('--val_interval', type=int, default=1,
                       help='Validation interval (epochs)')
    parser.add_argument('--viz_interval', type=int, default=5,
                       help='Visualization interval (epochs)')
    
    args = parser.parse_args()
    
    # Create config dictionary from arguments
    config = {
        'resnet': {
            'model_name': args.resnet_model,
            'target_layer': args.target_layer,
            'pretrained': args.pretrained
        },
        'sparse_dict': {
            'dict_size': args.dict_size,
            'sparsity_coef': args.sparsity_coef
        },
        'data': {
            'imagenet_path': args.imagenet_path,
            'many00_path': args.many00_path,
            'num_workers': args.num_workers
        },
        'training': {
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'log_interval': args.log_interval,
            'val_interval': args.val_interval,
            'viz_interval': args.viz_interval
        }
    }
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb only if requested
    if args.use_wandb:
        wandb.init(project="sparse-dictionary-learning", name=args.experiment_name, config=config)
        print("Weights & Biases logging enabled")
    else:
        print("Weights & Biases logging disabled")
    
    # Setup models
    resnet_extractor = ResNetFeatureExtractor(
        model_name=config['resnet']['model_name'],
        layer_name=config['resnet']['target_layer'],
        pretrained=config['resnet']['pretrained']
    ).to(device)
    
    # Get feature dimensions
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        dummy_features = resnet_extractor(dummy_input)
        feature_dim = dummy_features.shape[1]  # Should be 2048 for ResNet50 layer4
    
    print(f"Feature dimension: {feature_dim}")
    
    # Initialize sparse dictionary
    if args.use_subset:
        # Experiment 2: Use only 64 neurons
        input_dim = 64
        print("Running subset experiment with 64 neurons")
    else:
        # Experiment 1: Use all neurons
        input_dim = feature_dim
        print(f"Running full experiment with {feature_dim} neurons")
    
    sparse_dict = SparseDictionary(
        input_dim=input_dim,
        dict_size=config['sparse_dict']['dict_size'],
        sparsity_coef=config['sparse_dict']['sparsity_coef']
    ).to(device)
    
    # Setup data loaders
    if args.use_imagenet:
        print("Using ImageNet dataset")
        imagenet_loader = ImageNetDataLoader(
            root_path=config['data']['imagenet_path'],
            batch_size=config['training']['batch_size'],
            num_workers=config['data']['num_workers']
        )
        train_loader = imagenet_loader.get_train_loader()
        val_loader = imagenet_loader.get_val_loader()
    else:
        print("Using Many00 dataset")
        many00_loader = Many00DataLoader(
            root_path=config['data']['many00_path'],
            batch_size=config['training']['batch_size'],
            num_workers=config['data']['num_workers']
        )
        
        # For Many00, we'll use the same loader for both train and val
        # You can modify this if you have separate train/val splits
        train_loader = many00_loader.get_dataloader()
        val_loader = many00_loader.get_dataloader()  # Using same for now
    
    # Optimizer
    optimizer = optim.Adam(sparse_dict.parameters(), lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        sparse_dict.train()
        total_loss = 0
        total_recon_loss = 0
        total_sparsity_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["num_epochs"]}')
        
        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            
            # Extract features using ResNet
            with torch.no_grad():
                features = resnet_extractor(images)  # [batch_size, feature_dim]
            
            # For subset experiment, randomly sample 64 neurons
            if args.use_subset:
                indices = torch.randperm(feature_dim)[:64]
                features = features[:, indices]
            
            # Forward pass through sparse dictionary
            optimizer.zero_grad()
            reconstructed, activations = sparse_dict(features)
            
            # Compute losses
            recon_loss = compute_reconstruction_loss(features, reconstructed)
            sparsity_loss = compute_sparsity_loss(activations, config['sparse_dict']['sparsity_coef'])
            total_batch_loss = recon_loss + sparsity_loss
            
            # Backward pass
            total_batch_loss.backward()
            optimizer.step()
            
            # Update statistics
            total_loss += total_batch_loss.item()
            total_recon_loss += recon_loss.item()
            total_sparsity_loss += sparsity_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'Recon': f'{recon_loss.item():.4f}',
                'Sparse': f'{sparsity_loss.item():.4f}'
            })
            
            # Log to wandb
            if batch_idx % config['training']['log_interval'] == 0 and args.use_wandb:
                wandb.log({
                    'batch_loss': total_batch_loss.item(),
                    'batch_recon_loss': recon_loss.item(),
                    'batch_sparsity_loss': sparsity_loss.item(),
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
        
        # Epoch statistics
        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_sparsity_loss = total_sparsity_loss / len(train_loader)
        
        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Recon={avg_recon_loss:.4f}, Sparse={avg_sparsity_loss:.4f}')
        
        # Validation
        if epoch % config['training']['val_interval'] == 0:
            val_loss = validate(sparse_dict, resnet_extractor, val_loader, device, args.use_subset)
            scheduler.step(val_loss)
            
            if args.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': avg_loss,
                    'train_recon_loss': avg_recon_loss,
                    'train_sparsity_loss': avg_sparsity_loss,
                    'val_loss': val_loss
                })
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(sparse_dict, optimizer, epoch, val_loss, args.experiment_name)
        
        # Visualize dictionary features periodically
        if epoch % config['training']['viz_interval'] == 0:
            visualize_dictionary_features(sparse_dict, epoch, args.experiment_name)
    
    print("Training completed!")
    
    # Run post-training analysis for category activations
    print("\nStarting post-training category activation analysis...")
    try:
        # Use validation loader for analysis (or create a separate analysis loader)
        analysis_loader = val_loader
        
        # Run comprehensive analysis
        category_activations, category_counts = run_post_training_analysis(
            sparse_dict=sparse_dict,
            resnet_extractor=resnet_extractor, 
            dataloader=analysis_loader,
            device=device,
            experiment_name=args.experiment_name,
            use_subset=args.use_subset
        )
        
        # Log category analysis to wandb if enabled
        if args.use_wandb:
            wandb.log({
                'analysis/num_categories': len(category_counts),
                'analysis/total_samples_analyzed': sum(category_counts.values()),
            })
            
            # Log per-category sample counts
            for category, count in category_counts.items():
                wandb.log({f'analysis/category_samples/{category}': count})
        
        print(f"Category analysis completed for {len(category_counts)} categories")
        
    except Exception as e:
        print(f"Error during post-training analysis: {e}")
        print("Continuing without analysis...")
    
    if args.use_wandb:
        wandb.finish()
    
    # Restore original working directory
    os.chdir(original_cwd)
    print(f"Restored working directory to {original_cwd}")


def validate(sparse_dict, resnet_extractor, val_loader, device, use_subset):
    """
    Evaluates the sparse dictionary model on a validation dataset.

    This function sets the model to evaluation mode, iterates through the
    validation data loader, extracts features, performs reconstruction using
    the sparse dictionary, and calculates the combined reconstruction and
    sparsity loss.

    Args:
        sparse_dict (SparseDictionary): The sparse dictionary model to evaluate.
        resnet_extractor (ResNetFeatureExtractor): The ResNet model used for
            feature extraction.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): The device (CPU or CUDA) to perform computations on.
        use_subset (bool): If True, a random subset of 64 features is used,
            replicating one of the experiment conditions.

    Returns:
        float: The average validation loss (reconstruction + sparsity) over
               the entire validation set.
    """
    sparse_dict.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            features = resnet_extractor(images)
            
            if use_subset:
                indices = torch.randperm(features.shape[1])[:64]
                features = features[:, indices]
            
            reconstructed, activations = sparse_dict(features)
            recon_loss = compute_reconstruction_loss(features, reconstructed)
            sparsity_loss = compute_sparsity_loss(activations, sparse_dict.sparsity_coef)
            total_val_loss += (recon_loss + sparsity_loss).item()
    
    return total_val_loss / len(val_loader)


def save_checkpoint(model, optimizer, epoch, loss, experiment_name):
    """
    Saves the model and optimizer states to a checkpoint file.

    The checkpoint includes the current epoch, the model's state dictionary,
    the optimizer's state dictionary, and the loss at the time of saving.
    Checkpoints are saved in a 'checkpoints' directory, named with the
    experiment name and epoch number.

    Args:
        model (nn.Module): The model whose state needs to be saved.
        optimizer (optim.Optimizer): The optimizer whose state needs to be saved.
        epoch (int): The current epoch number.
        loss (float): The loss value at which the checkpoint is being saved
                      (typically validation loss).
        experiment_name (str): A name for the current experiment, used in the
                               checkpoint filename.

    Side Effects:
        Creates a directory named 'checkpoints' if it doesn't already exist.
        Writes a PyTorch checkpoint file (.pth) to the 'checkpoints' directory.
        Prints a confirmation message with the path to the saved checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = f'checkpoints/{experiment_name}_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved: {checkpoint_path}')


if __name__ == '__main__':
    main()