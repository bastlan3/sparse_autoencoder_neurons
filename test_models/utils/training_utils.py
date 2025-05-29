"""
Training utilities for sparse dictionary learning.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple


def compute_reconstruction_loss(original: torch.Tensor, reconstructed: torch.Tensor, 
                               loss_type: str = 'mse') -> torch.Tensor:
    """
    Compute reconstruction loss between original and reconstructed features.
    
    Args:
        original (torch.Tensor): Original features [batch_size, feature_dim]
        reconstructed (torch.Tensor): Reconstructed features [batch_size, feature_dim]
        loss_type (str): Type of loss ('mse', 'l1', 'huber')
        
    Returns:
        torch.Tensor: Reconstruction loss (scalar)
    """
    if loss_type == 'mse':
        loss = F.mse_loss(reconstructed, original)
    elif loss_type == 'l1':
        loss = F.l1_loss(reconstructed, original)
    elif loss_type == 'huber':
        loss = F.huber_loss(reconstructed, original)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    return loss


def compute_sparsity_loss(activations: torch.Tensor, sparsity_coef: float = 1e-3,
                         sparsity_type: str = 'l1') -> torch.Tensor:
    """
    Compute sparsity regularization loss.
    
    Args:
        activations (torch.Tensor): Sparse activations [batch_size, dict_size]
        sparsity_coef (float): Sparsity regularization coefficient
        sparsity_type (str): Type of sparsity regularization ('l1', 'l0_approx', 'entropy', 'target_sparsity')
        
    Returns:
        torch.Tensor: Sparsity loss (scalar)
    """
    if sparsity_type == 'l1':
        # Standard L1 sparsity
        sparsity_loss = sparsity_coef * torch.mean(torch.abs(activations))
    
    elif sparsity_type == 'l0_approx':
        # Smooth approximation to L0 norm using sigmoid
        # Approximates number of active neurons
        sigmoid_activations = torch.sigmoid(activations * 10)  # Sharp sigmoid
        sparsity_loss = sparsity_coef * torch.mean(sigmoid_activations)
    
    elif sparsity_type == 'entropy':
        # Entropy regularization to encourage peaky distributions
        probs = F.softmax(activations, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        sparsity_loss = -sparsity_coef * torch.mean(entropy)  # Negative because we want low entropy
    
    elif sparsity_type == 'target_sparsity':
        # Target a specific number of active features (1-4 per sample)
        target_active = 2.5  # Target 2-3 active features on average
        active_per_sample = torch.sum(activations > 0, dim=1).float()
        sparsity_loss = sparsity_coef * torch.mean((active_per_sample - target_active) ** 2)
    
    else:
        raise ValueError(f"Unsupported sparsity type: {sparsity_type}")
    
    return sparsity_loss


def compute_orthogonality_loss(dictionary_vectors: torch.Tensor, 
                             ortho_coef: float = 1e-4) -> torch.Tensor:
    """
    Compute orthogonality regularization for dictionary vectors.
    Encourages dictionary atoms to be orthogonal to each other.
    
    Args:
        dictionary_vectors (torch.Tensor): Dictionary matrix [input_dim, dict_size]
        ortho_coef (float): Orthogonality regularization coefficient
        
    Returns:
        torch.Tensor: Orthogonality loss (scalar)
    """
    # Normalize dictionary vectors
    normalized_dict = F.normalize(dictionary_vectors, dim=0)
    
    # Compute gram matrix (correlations between dictionary atoms)
    gram_matrix = torch.mm(normalized_dict.t(), normalized_dict)
    
    # We want the gram matrix to be close to identity
    identity = torch.eye(gram_matrix.size(0), device=gram_matrix.device)
    
    # Penalize off-diagonal elements
    off_diagonal_loss = torch.sum((gram_matrix - identity) ** 2) - torch.sum((torch.diag(gram_matrix) - 1) ** 2)
    
    return ortho_coef * off_diagonal_loss


def compute_total_loss(original: torch.Tensor, reconstructed: torch.Tensor,
                      activations: torch.Tensor, dictionary_vectors: torch.Tensor,
                      loss_config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Compute total loss with all components.
    
    Args:
        original: Original features
        reconstructed: Reconstructed features  
        activations: Sparse activations
        dictionary_vectors: Dictionary matrix
        loss_config: Configuration dictionary with loss weights and types
        
    Returns:
        Dictionary containing all loss components and total loss
    """
    losses = {}
    
    # Reconstruction loss
    recon_loss = compute_reconstruction_loss(
        original, reconstructed, 
        loss_type=loss_config.get('recon_loss_type', 'mse')
    )
    losses['reconstruction'] = recon_loss
    
    # Sparsity loss - use higher coefficient and target sparsity by default
    sparsity_loss = compute_sparsity_loss(
        activations,
        sparsity_coef=loss_config.get('sparsity_coef', 0.1),  # Increased default
        sparsity_type=loss_config.get('sparsity_type', 'target_sparsity')  # Changed default
    )
    losses['sparsity'] = sparsity_loss
    
    # Orthogonality loss (optional)
    if loss_config.get('use_orthogonality', False):
        ortho_loss = compute_orthogonality_loss(
            dictionary_vectors,
            ortho_coef=loss_config.get('ortho_coef', 1e-4)
        )
        losses['orthogonality'] = ortho_loss
    
    # Total loss
    total_loss = recon_loss + sparsity_loss
    if 'orthogonality' in losses:
        total_loss += losses['orthogonality']
    
    losses['total'] = total_loss
    
    return losses
