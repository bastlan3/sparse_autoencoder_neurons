import torch
import numpy as np
import time
import os
from tqdm import tqdm


class Trainer:
    """
    Trainer class for sparse autoencoder with dictionary learning.
    """
    def __init__(
        self,
        model,
        optimizer,
        device=None,
        save_dir='checkpoints',
        save_freq=10,
        patience=10,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The sparse autoencoder model
            optimizer: PyTorch optimizer
            device: Device to train on ('cuda' or 'cpu')
            save_dir: Directory to save model checkpoints
            save_freq: Frequency (in epochs) to save model checkpoints
            patience: Number of epochs to wait for improvement before early stopping
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.patience = patience
        
        self.model.to(self.device)
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon_loss': [],
            'val_recon_loss': [],
            'train_sparsity': [],
            'val_sparsity': []
        }
        
        # Best validation loss for early stopping
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        
    def train_epoch(self, dataloader):
        """
        Train the model for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            dict: Training metrics for the epoch
        """
        self.model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_sparsity = 0
        
        for batch_idx, data in enumerate(tqdm(dataloader, desc="Training")):
            data = data.to(self.device)
            
            # Forward pass
            reconstructed, encoded, sparsity_loss = self.model(data)
            reconstruction_loss = self.model.get_reconstruction_loss(data, reconstructed)
            loss = reconstruction_loss + sparsity_loss
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_recon_loss += reconstruction_loss.item()
            epoch_sparsity += sparsity_loss.item()
            
        # Calculate average metrics
        metrics = {
            'loss': epoch_loss / len(dataloader),
            'recon_loss': epoch_recon_loss / len(dataloader),
            'sparsity': epoch_sparsity / len(dataloader)
        }
        
        return metrics
    
    def validate(self, dataloader):
        """
        Validate the model on validation data.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_sparsity = 0
        
        with torch.no_grad():
            for data in tqdm(dataloader, desc="Validation"):
                data = data.to(self.device)
                
                # Forward pass
                reconstructed, encoded, sparsity_loss = self.model(data)
                reconstruction_loss = self.model.get_reconstruction_loss(data, reconstructed)
                loss = reconstruction_loss + sparsity_loss
                
                # Update metrics
                val_loss += loss.item()
                val_recon_loss += reconstruction_loss.item()
                val_sparsity += sparsity_loss.item()
                
        # Calculate average metrics
        metrics = {
            'loss': val_loss / len(dataloader),
            'recon_loss': val_recon_loss / len(dataloader),
            'sparsity': val_sparsity / len(dataloader)
        }
        
        return metrics
    
    def train(self, train_loader, val_loader=None, epochs=100):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train for
            
        Returns:
            dict: Training history
        """
        print(f"Training on {self.device}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Update training history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_recon_loss'].append(train_metrics['recon_loss'])
            self.history['train_sparsity'].append(train_metrics['sparsity'])
            
            # Validation if a validation loader is provided
            if val_loader:
                val_metrics = self.validate(val_loader)
                
                # Update validation history
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_recon_loss'].append(val_metrics['recon_loss'])
                self.history['val_sparsity'].append(val_metrics['sparsity'])
                
                # Check for early stopping
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.epochs_no_improve = 0
                    self.save_model(os.path.join(self.save_dir, 'best_model.pt'))
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve >= self.patience:
                        print(f"Early stopping after {epoch+1} epochs")
                        break
                
                val_log = f", Val Loss: {val_metrics['loss']:.4f}, Val Recon: {val_metrics['recon_loss']:.4f}, Val Sparsity: {val_metrics['sparsity']:.4f}"
            else:
                val_log = ""
            
            # Save model checkpoint
            if (epoch + 1) % self.save_freq == 0:
                self.save_model(os.path.join(self.save_dir, f'model_epoch_{epoch+1}.pt'))
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs}, Time: {epoch_time:.2f}s, "
                  f"Train Loss: {train_metrics['loss']:.4f}, Train Recon: {train_metrics['recon_loss']:.4f}, "
                  f"Train Sparsity: {train_metrics['sparsity']:.4f}{val_log}")
        
        # Save final model
        self.save_model(os.path.join(self.save_dir, 'final_model.pt'))
        
        return self.history
    
    def save_model(self, path):
        """
        Save model checkpoint.
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load model checkpoint.
        
        Args:
            path (str): Path to the saved model
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Model loaded from {path}")