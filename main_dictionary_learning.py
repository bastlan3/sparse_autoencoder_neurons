import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models, transforms

import numpy as np

from sklearn.decomposition import PCA
import pandas as pd
import pickle
import torch 
import torch.nn as nn
import torch.optim as optim # Added optimizer import
import gc
import matplotlib.pyplot as plt

import os
import sys

sys.path.append('../neural_switch/')
from multiple_pre_proc import process
# It's good practice to group imports from the same package or module group
# Also, os is imported twice, which is redundant but harmless.
# We'll keep one os import.
import os 
from sklearn.linear_model import LinearRegression

from torchvision import transforms # Already imported above, but kept for clarity of original structure
from PIL import Image
import pathlib # Already imported above

# Import the refactored class
# Assuming original_script.py and data_handling.py are in the same directory (refactored_code)
# If data_handling.py is meant to be a module, it might need an __init__.py in the directory
# and the import might be `from .data_handling import MultiSessionConcatDataset` if run as part of a package.
# For now, using direct import as if it's in PYTHONPATH.
from data.neural_data_loader import MultiSessionConcatDataset


# Removed TCA import as it's not used in this part
# from TCA import run_tca_analysis


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure this path is correct relative to where original_script.py is executed from.
# If original_script.py is in refactored_code, then './../data/Pa220227_0527.txt'
# means it looks for 'data/Pa220227_0527.txt' in the parent of 'refactored_code'.
# Now that the script is at the root, the path is direct.
try:
    lst_sess_file_path = './data/Pa220227_0527.txt' # Adjusted for script at root
    if not os.path.exists(lst_sess_file_path):
        raise FileNotFoundError(f"Session list file not found at {lst_sess_file_path}")
    lst_sess = [line.strip() for line in  open(lst_sess_file_path, 'r')]
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the session list file path is correct.")
    lst_sess = [] # Fallback to empty list


# --- Dictionary Learning Model ---

# Define the model
class DictionaryLearner(nn.Module):
    def __init__(self, input_dim, n_components):
        super(DictionaryLearner, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_components),
            nn.ReLU()
        )
        self.decoder = nn.Linear(n_components, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Instantiate the dataset
if lst_sess: # Only instantiate if session list was loaded
    dataset = MultiSessionConcatDataset(lst_sess, process, images = False)
else:
    dataset = None # Or handle error appropriately
    print("Warning: Dataset could not be instantiated as session list is empty.")

# Model parameters
n_components = 4000 # Example: Set the size of the dictionary (higher dimension)
learning_rate = 1e-4
num_epochs = 100 # Adjust as needed
batch_size = 2000 # Adjust based on memory availability
sparsity_lambda = 1e-3  # Adjust this value to control sparsity strength
alpha_sparsity_tanh = 5.0 # New hyperparameter for tanh sparsity

# DataLoader as before
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True if device.type == "cuda" else False,
    drop_last=False
)

# Check size of the first output of the dataloader
input_dim = 1920# 600 * 64 = 38400

# Instantiate model, loss, and optimizer
model_neur = DictionaryLearner(input_dim, n_components).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model_neur.parameters(), lr=learning_rate)



print("Starting training...")
# Training loop
torch.cuda.empty_cache()
gc.collect()
for epoch in range(100):
    epoch_loss = 0.0
    counter = 0
    for batch_data in dataloader:
        inputs = batch_data.float().to(device).reshape(batch_data.shape[0],-1)  # Move batch to GPU here
        # Forward pass
        encoded = model_neur.encoder(inputs)
        outputs = model_neur.decoder(encoded)
        recon_loss = criterion(outputs, inputs) # Reconstruction loss
        activity_measure = torch.tanh(alpha_sparsity_tanh * encoded)
        num_active_per_sample = torch.sum(activity_measure, dim=1) # Sum over components for each sample
        sparsity_penalty_val = torch.mean(num_active_per_sample) # Average over batch
        loss = recon_loss + sparsity_penalty_val

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * inputs.size(0)
        counter += 1
    epoch_loss /= len(dataset) # Calculate average loss for the epoch
    if (epoch ) % 10 == 0: # Print loss every 10 epochs
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        sparsity = (torch.linalg.norm(encoded, ord=10, dim=1) < 1e-3).float().mean().item()
        print(f'Sample encoding sparsity (fraction of |x| < 1e-3): {sparsity:.4f}')

print("Training finished.")
"""
# Evaluate reconstruction performance on a small sample from the training set
with torch.no_grad():
    sample_indices = np.random.choice(len(dataset), size=min(1000, len(dataset)), replace=False)
    sample = torch.stack([torch.tensor(dataset[i]) for i in sample_indices])
    sample = sample.float().to(device).reshape(sample.shape[0], -1)
    reconstructed = model(sample)
    mse_loss = criterion(reconstructed, sample).item()
    print(f'Sample reconstruction MSE loss: {mse_loss:.6f}')
    
    # Compute R^2 score
    y_true = sample.cpu().numpy().reshape(-1)
    y_pred = reconstructed.cpu().numpy().reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_score = 1 - ss_res / ss_tot
    print(f'Sample reconstruction R^2 score: {r2_score:.4f}')
    # Measure sparsity: fraction of (near-)zero elements in encoded representation
    encoded_sample = model.encoder(sample)
    sparsity = (torch.abs(encoded_sample) < 1e-3).float().mean().item()
    print(f'Sample encoding sparsity (fraction of |x| < 1e-3): {sparsity:.4f}')
# You can now use the trained model for analysis, e.g., get the encoded representations:
# encoded_data = model.encoder(sample)
# Or save the model:
# torch.save(model.state_dict(), 'dictionary_learner_model.pth')
"""
# Model parameters
n_components = 4000 # Example: Set the size of the dictionary (higher dimension)
learning_rate = 1e-4
num_epochs = 100 # Adjust as needed
batch_size = 100 # Adjust based on memory availability
sparsity_lambda = 1e-3  # Adjust this value to control sparsity strength

dataset = MultiSessionConcatDataset(lst_sess, process, images = True)

# DataLoader as before
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True if device.type == "cuda" else False,
    drop_last=False
)

# Load a pretrained ResNet50 and modify the final layers
resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Identity()  # Remove the final classification layer
resnet = resnet.to(device)
resnet.eval()  # Set to eval mode for feature extraction

# Feature extractor: outputs penultimate layer activations
def extract_features(images):
    with torch.no_grad():
        x = resnet(images)
    return x

# Add regression weights as the last layer of the model
class ResNetWithRegression(nn.Module):
    def __init__(self, base_model, reg):
        super().__init__()
        self.base = base_model
        self.regression = nn.Linear(reg.coef_.shape[1], reg.coef_.shape[0], bias=True)
        # Set weights and bias from sklearn regression
        self.regression.weight.data = torch.tensor(reg.coef_, dtype=torch.float32)
        self.regression.bias.data = torch.tensor(reg.intercept_, dtype=torch.float32)
        # Freeze all layers except the last two (regression layer and penultimate layer of base model)
        for name, param in self.base.named_parameters():
            param.requires_grad = False
        # Unfreeze the last two layers: regression and the last layer of base model
        for param in self.base.layer4.parameters():
            param.requires_grad = True
        for param in self.regression.parameters():
            param.requires_grad = True

    def forward(self, x):
        feats = self.base(x)
        return self.regression(feats)
    

torch.cuda.empty_cache()
gc.collect()
# Collect features and neural targets for linear regression
model_path = 'resnet_with_regression.pth'
reg_path = 'linear_regression_model.pkl'

if pathlib.Path(model_path).exists() and pathlib.Path(reg_path).exists():
    print("Loading saved regression model and features...")
    with open(reg_path, 'rb') as f:
        reg = pickle.load(f)
    
    # Check input and output size of regression
    print("Regression input size (features):", reg.coef_.shape[1] if reg.coef_.ndim > 1 else reg.coef_.shape[0])
    print("Regression output size (targets):", reg.coef_.shape[0])
    resnet_with_reg = ResNetWithRegression(resnet, reg).to(device)
    resnet_with_reg.load_state_dict(torch.load(model_path, map_location=device))
    resnet_with_reg.eval()
    print("Loaded pretrained regression model.")
    # Optionally skip feature extraction below if already loaded
else:
    print("No saved model found, extracting features...")
    all_features = []
    all_targets = []
    for neural_data, img_paths in dataloader:
        imgs = img_paths.to(device).float()  # Move images to GPU
        features = extract_features(imgs).cpu().numpy()
        neural_targets = neural_data.float().reshape(imgs.size(0), -1).cpu().numpy()
        all_features.append(features)
        all_targets.append(neural_targets)
        torch.cuda.empty_cache()
        gc.collect()
    print('imgs loaded ')
    torch.cuda.empty_cache()
    gc.collect()
    X = np.concatenate(all_features, axis=0)
    Y = np.concatenate(all_targets, axis=0)
    print(X.shape, Y.shape)
    # Fit linear regression from features to neural data
    reg = LinearRegression()
    reg.fit(X, Y)
    print(f"Linear regression R^2: {reg.score(X, Y):.4f}")
# Save the trained regression model and features



resnet_with_reg = ResNetWithRegression(resnet, reg).to(device)
resnet_with_reg.eval()

with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(reg, f)

torch.save(resnet_with_reg.state_dict(), 'resnet_with_regression.pth')

# Check size of the first output of the dataloader
input_dim = 1920# 600 * 64 = 38400
model = DictionaryLearner(input_dim, n_components).to(device)
# Instantiate model, loss, and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



print("Starting training...")
# Training loop

for epoch in range(num_epochs):
    epoch_loss = 0.0
    counter = 0
    torch.cuda.empty_cache()
    gc.collect()
    for batch_data in dataloader:
        inputs = resnet_with_reg(batch_data[1].to(device).float())  # Move batch to GPU here
        true_input = batch_data[0].float().to(device).reshape(batch_data[0].shape[0],-1)  # Move batch to GPU here
        # Forward pass
        encoded = model_neur.encoder(inputs)
        true_encoded = model.encoder(true_input)
        recon_loss = criterion(encoded, true_encoded) # Reconstruction loss
        sparsity_loss = sparsity_lambda * torch.mean(torch.abs(encoded)) # L1 penalty
        model_recon_loss = criterion(inputs, true_input) # Reconstruction loss
        loss = recon_loss + sparsity_loss + model_recon_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * inputs.size(0)
        counter += 1
        torch.cuda.empty_cache()
        gc.collect()
    epoch_loss /= len(dataset) # Calculate average loss for the epoch
    if (epoch ) % 10 == 0: # Print loss every 10 epochs
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
