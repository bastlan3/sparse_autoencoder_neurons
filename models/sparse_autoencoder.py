import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with linear encoding and decoding layers.
    
    Attributes:
        input_dim (int): Input dimension (neurons * time)
        embedding_dim (int): Dimension of the embedding layer (larger than input_dim)
        sparsity_weight (float): Weight for the L1 sparsity constraint
    """
    def __init__(self, input_dim, embedding_dim, sparsity_weight=1e-4):
        """
        Initialize the sparse autoencoder.
        
        Args:
            input_dim (int): Input dimension (neurons * time)
            embedding_dim (int): Dimension of the embedding layer (larger than input_dim)
            sparsity_weight (float): Weight for the L1 sparsity constraint
        """
        super(SparseAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.sparsity_weight = sparsity_weight
        
        # Encoder: input_dim -> embedding_dim
        self.encoder = nn.Linear(input_dim, embedding_dim, bias=True)
        
        # Decoder: embedding_dim -> input_dim
        self.decoder = nn.Linear(embedding_dim, input_dim, bias=True)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the weights for better convergence."""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
        
    def encode(self, x):
        """
        Encode the input data.
        
        Args:
            x (torch.Tensor): Input data

        Returns:
            torch.Tensor: Encoded representation
        """
        return self.encoder(x)
        
    def decode(self, z):
        """
        Decode the encoded representation.
        
        Args:
            z (torch.Tensor): Encoded representation
            
        Returns:
            torch.Tensor: Reconstructed data
        """
        return self.decoder(z)
        
    def forward(self, x):
        """
        Forward pass through the autoencoder.
        
        Args:
            x (torch.Tensor): Input data
            
        Returns:
            tuple: (reconstructed_x, encoded_z, sparsity_loss)
        """
        # Encode
        z = self.encode(x)
        
        # Apply non-linearity (optional based on your specific requirements)
        # z = F.relu(z)
        
        # Decode
        x_reconstructed = self.decode(z)
        
        # Calculate sparsity loss (L1 norm)
        sparsity_loss = self.sparsity_weight * torch.mean(torch.abs(z))
        
        return x_reconstructed, z, sparsity_loss
        
    def get_reconstruction_loss(self, x, x_reconstructed):
        """
        Calculate reconstruction loss (MSE).
        
        Args:
            x (torch.Tensor): Original input data
            x_reconstructed (torch.Tensor): Reconstructed data
            
        Returns:
            torch.Tensor: MSE loss
        """
        return F.mse_loss(x_reconstructed, x, reduction='mean')
    
    def get_total_loss(self, x, x_reconstructed, sparsity_loss):
        """
        Calculate total loss (reconstruction + sparsity).
        
        Args:
            x (torch.Tensor): Original input data
            x_reconstructed (torch.Tensor): Reconstructed data
            sparsity_loss (torch.Tensor): Sparsity loss term
            
        Returns:
            torch.Tensor: Total loss
        """
        reconstruction_loss = self.get_reconstruction_loss(x, x_reconstructed)
        return reconstruction_loss + sparsity_loss