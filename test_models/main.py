 
#!/usr/bin/env python3
"""
Main training script for sparse dictionary learning on ResNet50 features.
Recreates findings from "Towards Monosemanticity" paper.
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
from utils.visualization import visualize_dictionary_features


def main():
    parser = argparse.ArgumentParser(description='Sparse Dictionary Learning on ResNet50')
    parser.add_argument('--config', type=str, default='configs/default.json', 
                       help='Path to configuration file')
    parser.add_argument('--experiment_name', type=str, default='sparse_dict_resnet50',
                       help='Name for the experiment')
    parser.add_argument('--use_subset', action='store_true',
                       help='Use only 64 neurons subset experiment')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    wandb.init(project="sparse-dictionary-learning", name=args.experiment_name, config=config)
    
    # Setup models
    resnet_extractor = ResNetFeatureExtractor(
        model_name=config['resnet']['model_name'],
        layer_name=config['resnet']['target_layer'],
        pretrained=True
    ).to(device)
    
    # Get feature dimensions
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        dummy_features = resnet_extractor(dummy_input)
        feature_dim = dummy_features.shape[1]  # Should be 2048 for ResNet50 penultimate layer
    
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
    imagenet_loader = ImageNetDataLoader(
        root_path=config['data']['imagenet_path'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    
    many00_loader = Many00DataLoader(
        root_path=config['data']['many00_path'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        # Additional params will be specified in the loader implementation
    )
    
    # Combine data loaders or alternate between them
    train_loader = imagenet_loader.get_train_loader()
    val_loader = imagenet_loader.get_val_loader()
    
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
            if batch_idx % config['training']['log_interval'] == 0:
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
    wandb.finish()


def validate(sparse_dict, resnet_extractor, val_loader, device, use_subset):
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