"""
ImageNet data loader for sparse dictionary learning experiments.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
from typing import Optional, Tuple


class ImageNetDataLoader:
    """
    Data loader for ImageNet dataset.
    Handles both training and validation splits with appropriate preprocessing.
    """
    
    def __init__(self, 
                 root_path: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 shuffle_train: bool = True,
                 image_size: int = 224,
                 normalize: bool = True):
        """
        Initialize ImageNet data loader.
        
        Args:
            root_path (str): Path to ImageNet dataset root directory
                           Should contain 'train' and 'val' subdirectories
            batch_size (int): Batch size for data loading
            num_workers (int): Number of worker processes for data loading
            pin_memory (bool): Whether to use pinned memory
            shuffle_train (bool): Whether to shuffle training data
            image_size (int): Size to resize images to (square)
            normalize (bool): Whether to apply ImageNet normalization
        """
        self.root_path = root_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.image_size = image_size
        self.normalize = normalize
        
        # Verify dataset structure
        self._verify_dataset_structure()
        
        # Setup transforms
        self.train_transform = self._get_train_transforms()
        self.val_transform = self._get_val_transforms()
    
    def _verify_dataset_structure(self):
        """Verify that the ImageNet dataset has the expected structure."""
        train_path = os.path.join(self.root_path, 'train')
        val_path = os.path.join(self.root_path, 'val')
        
        if not os.path.exists(self.root_path):
            raise FileNotFoundError(f"Dataset root path not found: {self.root_path}")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found: {train_path}")
        
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation data not found: {val_path}")
        
        print(f"ImageNet dataset found at: {self.root_path}")
        print(f"Training data: {train_path}")
        print(f"Validation data: {val_path}")
    
    def _get_train_transforms(self):
        """Get training data transforms."""
        transform_list = [
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor()
        ]
        
        if self.normalize:
            # ImageNet normalization values
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        return transforms.Compose(transform_list)
    
    def _get_val_transforms(self):
        """Get validation data transforms."""
        transform_list = [
            transforms.Resize(256),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor()
        ]
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        return transforms.Compose(transform_list)
    
    def get_train_loader(self):
        """Get training data loader."""
        train_dataset = datasets.ImageFolder(
            root=os.path.join(self.root_path, 'train'),
            transform=self.train_transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True  # Important for consistent batch sizes
        )
        
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Number of training batches: {len(train_loader)}")
        
        return train_loader
    
    def get_val_loader(self):
        """Get validation data loader."""
        val_dataset = datasets.ImageFolder(
            root=os.path.join(self.root_path, 'val'),
            transform=self.val_transform
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
        
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Number of validation batches: {len(val_loader)}")
        
        return val_loader
    
    def get_class_names(self):
        """Get class names from the dataset."""
        train_dataset = datasets.ImageFolder(
            root=os.path.join(self.root_path, 'train'),
            transform=self.val_transform
        )
        return train_dataset.classes
    
    def get_sample_batch(self, split='train'):
        """Get a sample batch for testing purposes."""
        if split == 'train':
            loader = self.get_train_loader()
        else:
            loader = self.get_val_loader()
        
        for batch in loader:
            return batch  # Return first batch
    
    def compute_dataset_stats(self, num_samples=1000):
        """
        Compute mean and std statistics for the dataset.
        Useful for custom normalization.
        """
        print(f"Computing dataset statistics from {num_samples} samples...")
        
        # Use minimal transforms for statistics computation
        temp_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor()
        ])
        
        temp_dataset = datasets.ImageFolder(
            root=os.path.join(self.root_path, 'train'),
            transform=temp_transform
        )
        
        # Sample random indices
        indices = np.random.choice(len(temp_dataset), min(num_samples, len(temp_dataset)), replace=False)
        
        # Compute statistics
        mean = torch.zeros(3)
        std = torch.zeros(3)
        total_samples = 0
        
        for idx in indices:
            image, _ = temp_dataset[idx]
            mean += image.mean(dim=[1, 2])
            std += image.std(dim=[1, 2])
            total_samples += 1
        
        mean /= total_samples
        std /= total_samples
        
        print(f"Dataset statistics:")
        print(f"Mean: {mean.tolist()}")
        print(f"Std: {std.tolist()}")
        
        return mean.tolist(), std.tolist()


class ImageNetSubsetLoader(ImageNetDataLoader):
    """
    Data loader for a subset of ImageNet classes.
    Useful for faster experimentation.
    """
    
    def __init__(self, class_subset=None, max_samples_per_class=None, **kwargs):
        """
        Initialize ImageNet subset loader.
        
        Args:
            class_subset (list): List of class names to include
            max_samples_per_class (int): Maximum samples per class
            **kwargs: Arguments passed to parent class
        """
        self.class_subset = class_subset
        self.max_samples_per_class = max_samples_per_class
        super().__init__(**kwargs)
    
    def get_train_loader(self):
        """Get training loader with subset constraints."""
        full_dataset = datasets.ImageFolder(
            root=os.path.join(self.root_path, 'train'),
            transform=self.train_transform
        )
        
        if self.class_subset is not None or self.max_samples_per_class is not None:
            subset_dataset = self._create_subset(full_dataset)
        else:
            subset_dataset = full_dataset
        
        train_loader = DataLoader(
            subset_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        
        print(f"Subset training dataset size: {len(subset_dataset)}")
        return train_loader
    
    def _create_subset(self, dataset):
        """Create subset based on class and sample constraints."""
        indices = []
        class_to_idx = dataset.class_to_idx
        
        # Get target classes
        if self.class_subset is not None:
            target_classes = [class_to_idx[cls] for cls in self.class_subset if cls in class_to_idx]
        else:
            target_classes = list(range(len(dataset.classes)))
        
        # Count samples per class
        class_counts = {cls: 0 for cls in target_classes}
        
        for idx, (_, label) in enumerate(dataset.samples):
            if label in target_classes:
                if (self.max_samples_per_class is None or 
                    class_counts[label] < self.max_samples_per_class):
                    indices.append(idx)
                    class_counts[label] += 1
        
        return torch.utils.data.Subset(dataset, indices)