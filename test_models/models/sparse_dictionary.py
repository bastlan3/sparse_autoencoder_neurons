"""
Sparse Dictionary model implementation for monosemanticity research.
Based on the architecture described in "Towards Monosemanticity" paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseDictionary(nn.Module):
    """
    Sparse Dictionary Learning module with encoder-decoder architecture.
    
    Architecture:
    - Encoder: Linear layer + ReLU -> sparse high-dimensional space
    - Decoder: Linear layer -> reconstruction to original space
    """
    
    def __init__(self, input_dim, dict_size, sparsity_coef=1e-3, tie_weights=False):
        """
        Initialize Sparse Dictionary.
        
        Args:
            input_dim (int): Dimension of input features (e.g., 2048 for ResNet50)
            dict_size (int): Size of dictionary (sparse dimension, typically >> input_dim)
            sparsity_coef (float): Coefficient for sparsity regularization
            tie_weights (bool): Whether to tie encoder and decoder weights
        """
        super(SparseDictionary, self).__init__()
        
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.sparsity_coef = sparsity_coef
        self.tie_weights = tie_weights
        
        # Encoder: Projects to sparse dictionary space
        self.encoder = nn.Linear(input_dim, dict_size, bias=True)
        
        # Decoder: Projects back to original space
        if tie_weights:
            # Tied weights: decoder is transpose of encoder
            self.decoder = None
        else:
            self.decoder = nn.Linear(dict_size, input_dim, bias=False)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights with proper scaling."""
        # Initialize encoder weights
        nn.init.kaiming_normal_(self.encoder.weight, mode='fan_out', nonlinearity='relu')
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)
        
        # Initialize decoder weights
        if not self.tie_weights:
            nn.init.kaiming_normal_(self.decoder.weight, mode='fan_in', nonlinearity='linear')
        
        # If using tied weights, ensure decoder weights are normalized
        if self.tie_weights:
            with torch.no_grad():
                # Normalize columns of encoder weight matrix
                self.encoder.weight.data = F.normalize(self.encoder.weight.data, dim=0)
    
    def encode(self, x):
        """
        Encode input to sparse dictionary space.
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Sparse activations [batch_size, dict_size]
        """
        # Linear projection
        activations = self.encoder(x)
        
        # Apply ReLU for sparsity
        sparse_activations = F.relu(activations)
        
        return sparse_activations
    
    def decode(self, activations):
        """
        Decode sparse activations back to original space.
        
        Args:
            activations (torch.Tensor): Sparse activations [batch_size, dict_size]
            
        Returns:
            torch.Tensor: Reconstructed features [batch_size, input_dim]
        """
        if self.tie_weights:
            # Use transpose of encoder weights
            reconstructed = F.linear(activations, self.encoder.weight.t())
        else:
            reconstructed = self.decoder(activations)
        
        return reconstructed
    
    def forward(self, x):
        """
        Forward pass through sparse dictionary.
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
            
        Returns:
            tuple: (reconstructed_features, sparse_activations)
        """
        # Encode to sparse space
        sparse_activations = self.encode(x)
        
        # Decode back to original space
        reconstructed = self.decode(sparse_activations)
        
        return reconstructed, sparse_activations
    
    def get_dictionary_vectors(self):
        """
        Get the dictionary vectors (columns of decoder weight matrix).
        
        Returns:
            torch.Tensor: Dictionary vectors [input_dim, dict_size]
        """
        if self.tie_weights:
            return self.encoder.weight.data.t()
        else:
            return self.decoder.weight.data.t()
    
    def get_feature_activations(self, x, return_indices=False):
        """
        Get activations and optionally the indices of active features.
        
        Args:
            x (torch.Tensor): Input features
            return_indices (bool): Whether to return indices of active features
            
        Returns:
            torch.Tensor or tuple: Activations and optionally active indices
        """
        activations = self.encode(x)
        
        if return_indices:
            # Get indices where activations are non-zero
            active_indices = (activations > 0).nonzero(as_tuple=True)
            return activations, active_indices
        
        return activations
    
    def compute_sparsity_metrics(self, x):
        """
        Compute various sparsity metrics for analysis.
        
        Args:
            x (torch.Tensor): Input features
            
        Returns:
            dict: Dictionary containing sparsity metrics
        """
        with torch.no_grad():
            activations = self.encode(x)
            
            # L0 sparsity (fraction of non-zero activations)
            l0_sparsity = (activations > 0).float().mean().item()
            
            # L1 norm
            l1_norm = activations.abs().mean().item()
            
            # Max activation
            max_activation = activations.max().item()
            
            # Number of active features per sample
            active_per_sample = (activations > 0).sum(dim=1).float().mean().item()
            
            return {
                'l0_sparsity': l0_sparsity,
                'l1_norm': l1_norm,
                'max_activation': max_activation,
                'active_per_sample': active_per_sample,
                'total_dict_size': self.dict_size
            }


class AdaptiveSparseDictionary(SparseDictionary):
    """
    Adaptive version of Sparse Dictionary with learnable sparsity threshold.
    """
    
    def __init__(self, input_dim, dict_size, initial_threshold=0.1, **kwargs):
        super().__init__(input_dim, dict_size, **kwargs)
        
        # Learnable threshold parameter
        self.threshold = nn.Parameter(torch.tensor(initial_threshold))
    
    def encode(self, x):
        """Encode with adaptive threshold."""
        activations = self.encoder(x)
        
        # Apply adaptive threshold instead of ReLU
        sparse_activations = F.relu(activations - self.threshold)
        
        return sparse_activations


class TopKSparseDictionary(SparseDictionary):
    """
    Sparse Dictionary with Top-K sparsity constraint.
    """
    
    def __init__(self, input_dim, dict_size, k=None, **kwargs):
        super().__init__(input_dim, dict_size, **kwargs)
        self.k = k or max(1, dict_size // 100)  # Default to 1% of dictionary size
    
    def encode(self, x):
        """Encode with Top-K sparsity."""
        activations = self.encoder(x)
        
        # Apply ReLU first
        relu_activations = F.relu(activations)
        
        # Apply Top-K sparsity
        if self.training:
            # During training, use soft Top-K (keeps gradients)
            values, indices = torch.topk(relu_activations, self.k, dim=1)
            sparse_activations = torch.zeros_like(relu_activations)
            sparse_activations.scatter_(1, indices, values)
        else:
            # During inference, use hard Top-K
            values, indices = torch.topk(relu_activations, self.k, dim=1)
            sparse_activations = torch.zeros_like(relu_activations)
            sparse_activations.scatter_(1, indices, values)
        
        return sparse_activations 
