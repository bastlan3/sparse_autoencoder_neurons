import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class NeuralDataset(Dataset):
    """
    Dataset class for neural recordings with shape: [neurons, time, presentations]
    """
    def __init__(self, data_path=None, data=None, transform=None):
        """
        Initialize the neural dataset.
        
        Args:
            data_path (str, optional): Path to the data file (.npy or .npz)
            data (numpy.ndarray, optional): Data array if already loaded
            transform (callable, optional): Optional transform to be applied on a sample
        """
        if data is not None:
            self.data = data
        elif data_path is not None:
            self.load_data(data_path)
        else:
            raise ValueError("Either data or data_path must be provided")
            
        self.transform = transform
        self.n_neurons, self.time_steps, self.n_presentations = self.data.shape
        
    def load_data(self, data_path):
        """
        Load neural data from file.
        
        Args:
            data_path (str): Path to the data file
        """
        if data_path.endswith('.npy'):
            self.data = np.load(data_path)
        elif data_path.endswith('.npz'):
            loaded = np.load(data_path)
            # Assuming the data is stored under a key, adjust if needed
            self.data = loaded[list(loaded.keys())[0]]
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
            
        # Ensure data has shape [neurons, time, presentations]
        if len(self.data.shape) != 3:
            raise ValueError(f"Data should have 3 dimensions [neurons, time, presentations], got {self.data.shape}")
            
    def __len__(self):
        """Return the number of presentations in the dataset."""
        return self.n_presentations
        
    def __getitem__(self, idx):
        """
        Get a single presentation's neural activity.
        
        Args:
            idx (int): Index of the presentation
            
        Returns:
            torch.Tensor: Neural activity for the presentation with shape [neurons, time]
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get the neural data for the specified presentation
        sample = self.data[:, :, idx]
        
        # Reshape to [neurons * time] for the autoencoder input
        sample_flattened = sample.reshape(-1)
        
        # Convert to torch tensor
        sample_tensor = torch.from_numpy(sample_flattened).float()
        
        # Apply transforms if available
        if self.transform:
            sample_tensor = self.transform(sample_tensor)
            
        return sample_tensor


def get_neural_dataloader(data_path=None, data=None, batch_size=32, shuffle=True, num_workers=4):
    """
    Create a DataLoader for neural data.
    
    Args:
        data_path (str, optional): Path to the data file
        data (numpy.ndarray, optional): Data array if already loaded
        batch_size (int): Batch size for training
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of workers for data loading
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for the neural data
    """
    dataset = NeuralDataset(data_path=data_path, data=data)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataloader, dataset