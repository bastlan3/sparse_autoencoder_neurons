import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

def plot_training_history(history):
    """
    Plot the training history metrics.
    
    Args:
        history (dict): Dictionary containing training history
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot losses
    ax = axes[0]
    ax.plot(history['train_loss'], label='Train Loss', color='blue')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        ax.plot(history['val_loss'], label='Val Loss', color='red')
    ax.set_title('Total Loss Over Time')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Plot sparsity
    ax = axes[1]
    ax.plot(history['train_sparsity'], label='Train Sparsity', color='green')
    if 'val_sparsity' in history and len(history['val_sparsity']) > 0:
        ax.plot(history['val_sparsity'], label='Val Sparsity', color='orange')
    ax.set_title('Sparsity Over Time')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L1 Sparsity')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    return fig

def plot_reconstructions(model, data_loader, device, n_samples=5):
    """
    Plot original and reconstructed samples.
    
    Args:
        model: Trained autoencoder model
        data_loader: DataLoader containing samples
        device: Device to run inference on
        n_samples: Number of samples to visualize
    """
    model.eval()
    samples = []
    reconstructions = []
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= n_samples:
                break
                
            data = data.to(device)
            recon, _, _ = model(data)
            
            samples.append(data.cpu().numpy())
            reconstructions.append(recon.cpu().numpy())
    
    samples = np.array(samples).squeeze()
    reconstructions = np.array(reconstructions).squeeze()
    
    n_neurons = int(np.sqrt(samples.shape[1]))
    
    # Try to reshape to [neurons, time] format
    try:
        samples_reshaped = samples.reshape(samples.shape[0], n_neurons, -1)
        recons_reshaped = reconstructions.reshape(reconstructions.shape[0], n_neurons, -1)
        reshape_ok = True
    except:
        reshape_ok = False
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3 * n_samples))
    
    for i in range(n_samples):
        if n_samples == 1:
            ax1, ax2 = axes
        else:
            ax1, ax2 = axes[i]
            
        # Plot original
        if reshape_ok:
            im1 = ax1.imshow(samples_reshaped[i], aspect='auto', cmap='viridis')
        else:
            im1 = ax1.plot(samples[i])
        ax1.set_title(f"Original {i+1}")
        
        # Plot reconstruction
        if reshape_ok:
            im2 = ax2.imshow(recons_reshaped[i], aspect='auto', cmap='viridis')
        else:
            im2 = ax2.plot(reconstructions[i])
        ax2.set_title(f"Reconstruction {i+1}")
    
    plt.tight_layout()
    return fig

def plot_dictionary_elements(model, n_elements=20, figsize=(15, 10)):
    """
    Visualize dictionary elements (decoder weights).
    
    Args:
        model: Trained autoencoder model
        n_elements: Number of dictionary elements to visualize
        figsize: Figure size
    """
    decoder_weights = model.decoder.weight.detach().cpu().numpy()
    
    # Calculate importance of each dictionary element (L2 norm)
    importance = np.linalg.norm(decoder_weights, axis=0)
    top_indices = np.argsort(importance)[-n_elements:][::-1]
    
    # Get input dimensions
    input_dim = decoder_weights.shape[0]
    
    # Try to determine if we can reshape to 2D
    n_neurons = int(np.sqrt(input_dim))
    
    fig, axes = plt.subplots(4, 5, figsize=figsize)
    axes = axes.flatten()
    
    for i, idx in enumerate(top_indices):
        if i >= len(axes):
            break
            
        dict_element = decoder_weights[:, idx]
        
        # Try to reshape to 2D if possible
        try:
            dict_element_2d = dict_element.reshape(n_neurons, -1)
            axes[i].imshow(dict_element_2d, cmap='viridis', aspect='auto')
        except:
            axes[i].plot(dict_element)
            
        axes[i].set_title(f"Element {idx}")
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def plot_encoding_statistics(model, data_loader, device):
    """
    Plot statistics about the encoded representations.
    
    Args:
        model: Trained autoencoder model
        data_loader: DataLoader containing samples
        device: Device to run inference on
    """
    model.eval()
    all_encodings = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            _, encoded, _ = model(data)
            all_encodings.append(encoded.cpu().numpy())
    
    all_encodings = np.vstack(all_encodings)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot activation frequency (how often each feature is active)
    sparsity_threshold = 0.01  # Arbitrary threshold to consider a feature "active"
    activation_frequency = np.mean(np.abs(all_encodings) > sparsity_threshold, axis=0)
    sns.histplot(activation_frequency, kde=True, ax=axes[0])
    axes[0].set_title('Feature Activation Frequency')
    axes[0].set_xlabel('Frequency')
    axes[0].set_ylabel('Count')
    
    # Plot average activation magnitude
    avg_magnitude = np.mean(np.abs(all_encodings), axis=0)
    sns.histplot(avg_magnitude, kde=True, ax=axes[1])
    axes[1].set_title('Average Feature Magnitude')
    axes[1].set_xlabel('Magnitude')
    axes[1].set_ylabel('Count')
    
    # Plot sparsity per sample (what % of features are active)
    sparsity_per_sample = np.mean(np.abs(all_encodings) > sparsity_threshold, axis=1)
    sns.histplot(sparsity_per_sample, kde=True, ax=axes[2])
    axes[2].set_title('Encoding Sparsity per Sample')
    axes[2].set_xlabel('Active Features (%)')
    axes[2].set_ylabel('Count')
    
    plt.tight_layout()
    return fig, all_encodings