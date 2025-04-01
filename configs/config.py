"""
Configuration for Sparse Autoencoder for Neural Data
"""

# Data settings
data_config = {
    'batch_size': 32,
    'shuffle': True,
    'num_workers': 4,
    'train_val_split': 0.8,  # Percentage of data to use for training
}

# Model settings
model_config = {
    'embedding_dim': 1024,  # Larger than input_dim for overcomplete dictionary
    'sparsity_weight': 0.001,  # Weight for L1 sparsity regularization
}

# Training settings
training_config = {
    'epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'patience': 10,  # For early stopping
    'save_freq': 10,  # Save model every N epochs
    'save_dir': './sparse_autoencoder/checkpoints',
}

# Visualization settings
viz_config = {
    'n_reconstruction_samples': 5,  # Number of samples to visualize
    'n_dictionary_elements': 20,    # Number of dictionary elements to visualize
}