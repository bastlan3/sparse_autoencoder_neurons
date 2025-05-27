 
"""
Many00 custom dataset loader for sparse dictionary learning experiments.
This is a template/draft that needs to be customized based on your specific dataset structure.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import json
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Union


class Many00Dataset(Dataset):
    """
    Custom dataset class for Many00 dataset.
    
    IMPORTANT: This is a DRAFT implementation. You need to specify:
    1. The exact directory structure of your Many00 dataset
    2. How images are organized (subdirectories, file naming convention)
    3. Whether you have labels/annotations and their format
    4. Any metadata files (CSV, JSON, etc.)
    5. Image file extensions you want to support
    """
    
    def __init__(self, 
                 root_path: str,
                 transform: Optional[transforms.Compose] = None,
                 # MISSING INFORMATION - Please specify:
                 image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
                 recursive_search: bool = True,
                 annotation_file: Optional[str] = None,
                 metadata_file: Optional[str] = None,
                 class_mapping_file: Optional[str] = None,
                 subset_classes: Optional[List[str]] = None,
                 max_samples: Optional[int] = None):
        """
        Initialize Many00 dataset.
        
        Args:
            root_path (str): Path to Many00 dataset root directory
            transform: Image transformations to apply
            image_extensions: Tuple of supported image file extensions
            recursive_search: Whether to search subdirectories recursively
            annotation_file: Path to annotation file (if exists)
            metadata_file: Path to metadata file (if exists)  
            class_mapping_file: Path to class mapping file (if exists)
            subset_classes: List of specific classes to load (if applicable)
            max_samples: Maximum number of samples to load
            
        MISSING INFORMATION NEEDED:
        - What is the directory structure? E.g.:
          * many00/class1/image1.jpg, many00/class2/image2.jpg
          * many00/images/img1.jpg with separate labels file
          * many00/train/..., many00/val/...
        - Do you have labels? What format?
        - Are there any metadata files (CSV, JSON)?
        - What image formats are you using?
        - Do you need train/val splits?
        """
        self.root_path = root_path
        self.transform = transform
        self.image_extensions = image_extensions
        self.recursive_search = recursive_search
        self.annotation_file = annotation_file
        self.metadata_file = metadata_file
        self.subset_classes = subset_classes
        self.max_samples = max_samples
        
        # Initialize data structures
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.classes = []
        self.metadata = {}
        
        # Load dataset
        self._load_dataset()
        
        print(f"Many00 dataset loaded: {len(self.image_paths)} images, {len(self.classes)} classes")
    
    def _load_dataset(self):
        """
        Load the dataset based on the directory structure.
        
        THIS NEEDS TO BE CUSTOMIZED based on your specific dataset structure!
        """
        if not os.path.exists(self.root_path):
            raise FileNotFoundError(f"Dataset root path not found: {self.root_path}")
        
        # METHOD 1: Directory-based classes (like ImageFolder)
        # Uncomment and modify if your data is organized as: many00/class1/img1.jpg, many00/class2/img2.jpg
        self._load_directory_based()
        
        # METHOD 2: Single directory with annotation file
        # Uncomment if you have: many00/images/ + annotations.csv/json
        # self._load_with_annotations()
        
        # METHOD 3: Custom structure
        # Implement your own loading logic here
        # self._load_custom_structure()
        
        # Apply subset filtering if specified
        if self.subset_classes is not None:
            self._filter_by_classes()
        
        # Apply max samples limit if specified
        if self.max_samples is not None:
            self._limit_samples()
    
    def _load_directory_based(self):
        """Load dataset assuming directory-based class organization."""
        print("Loading directory-based dataset structure...")
        
        # Get all subdirectories as classes
        potential_classes = [d for d in os.listdir(self.root_path) 
                           if os.path.isdir(os.path.join(self.root_path, d))]
        potential_classes.sort()
        
        if not potential_classes:
            raise FileNotFoundError(
                f"No category subdirectories found in {self.root_path}. "
                "Many00 dataset expects images to be organized into subfolders, "
                "each representing a category."
            )
        
        # Multiple classes based on subdirectories
        self.classes = potential_classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_path = os.path.join(self.root_path, class_name)
            class_idx = self.class_to_idx[class_name]
            # Assuming images are directly in class_path, not further subdirectories within class_path
            # So, recursive_search for _load_images_from_directory might be better set to False
            # However, the current implementation of _load_images_from_directory handles both cases correctly.
            # If recursive_search is True (default), it walks starting from class_path.
            # If recursive_search is False, it lists files in class_path.
            # For the structure 'root/category/image.jpg', either will work.
            # Let's stick to the default recursive_search=True for flexibility,
            # as it doesn't harm the specified structure.
            self._load_images_from_directory(class_path, class_idx)
    
    def _load_images_from_directory(self, directory: str, class_idx: int):
        """Load all images from a directory."""
        if self.recursive_search:
            # Recursively search subdirectories
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(self.image_extensions):
                        image_path = os.path.join(root, file)
                        self.image_paths.append(image_path)
                        self.labels.append(class_idx)
        else:
            # Only search immediate directory
            for file in os.listdir(directory):
                if file.lower().endswith(self.image_extensions):
                    image_path = os.path.join(directory, file)
                    if os.path.isfile(image_path):
                        self.image_paths.append(image_path)
                        self.labels.append(class_idx)
    
    def _load_with_annotations(self):
        """
        Load dataset with separate annotation file.
        
        CUSTOMIZE THIS based on your annotation format!
        """
        print("Loading dataset with annotations...")
        
        if self.annotation_file is None:
            raise ValueError("Annotation file must be specified for this loading method")
        
        annotation_path = os.path.join(self.root_path, self.annotation_file)
        
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        
        # Load annotations based on file type
        if annotation_path.endswith('.csv'):
            annotations = pd.read_csv(annotation_path)
            # CUSTOMIZE: Specify column names for image paths and labels
            # Example: annotations should have columns 'image_path' and 'label'
            # self.image_paths = [os.path.join(self.root_path, 'images', path) 
            #                    for path in annotations['image_path'].tolist()]
            # self.labels = annotations['label'].tolist()
            
        elif annotation_path.endswith('.json'):
            with open(annotation_path, 'r') as f:
                annotations = json.load(f)
            # CUSTOMIZE: Parse JSON structure
            # Example format: {"images": [{"path": "img1.jpg", "label": "class1"}, ...]}
            
        # Create class mappings
        unique_labels = list(set(self.labels))
        unique_labels.sort()
        self.classes = unique_labels
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Convert string labels to indices if needed
        if isinstance(self.labels[0], str):
            self.labels = [self.class_to_idx[label] for label in self.labels]
    
    def _load_custom_structure(self):
        """
        Custom loading method for specific dataset structure.
        
        IMPLEMENT THIS based on your exact requirements!
        """
        print("Loading custom dataset structure...")
        
        # Example: If you have a specific naming convention or structure
        # Implement your custom logic here
        
        pass
    
    def _filter_by_classes(self):
        """Filter dataset to include only specified classes."""
        if not self.subset_classes:
            return
        
        # Get indices of target classes
        target_indices = [self.class_to_idx.get(cls) for cls in self.subset_classes 
                         if cls in self.class_to_idx]
        
        if not target_indices:
            raise ValueError(f"None of the specified classes found: {self.subset_classes}")
        
        # Filter images and labels
        filtered_paths = []
        filtered_labels = []
        
        for path, label in zip(self.image_paths, self.labels):
            if label in target_indices:
                filtered_paths.append(path)
                filtered_labels.append(label)
        
        self.image_paths = filtered_paths
        self.labels = filtered_labels
        
        print(f"Filtered to {len(self.image_paths)} images from {len(self.subset_classes)} classes")
    
    def _limit_samples(self):
        """Limit the number of samples."""
        if self.max_samples and len(self.image_paths) > self.max_samples:
            # Randomly sample
            indices = np.random.choice(len(self.image_paths), self.max_samples, replace=False)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
            print(f"Limited to {self.max_samples} samples")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        image_path = self.image_paths[idx]
        label = self.labels[idx] if self.labels else 0
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image in case of error
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """Get the distribution of classes in the dataset."""
        if not self.labels:
            return {}
        
        class_counts = {}
        for label in self.labels:
            class_name = self.classes[label] if label < len(self.classes) else f"class_{label}"
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return class_counts


class Many00DataLoader:
    """
    Data loader wrapper for Many00 dataset.
    """
    
    def __init__(self,
                 root_path: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 shuffle: bool = True,
                 image_size: int = 224,
                 normalize: bool = True,
                 # Dataset-specific parameters - CUSTOMIZE THESE
                 **dataset_kwargs):
        """
        Initialize Many00 data loader.
        
        MISSING INFORMATION - Please specify:
        1. root_path: Path to your many00 dataset
        2. Any additional dataset_kwargs specific to your data structure
        
        Example usage:
        loader = Many00DataLoader(
            root_path='/path/to/many00',
            batch_size=32,
            annotation_file='annotations.csv',  # if you have annotations
            recursive_search=True,  # if images are in subdirectories
            subset_classes=['class1', 'class2'],  # if you want specific classes
        )
        """
        self.root_path = root_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.image_size = image_size
        self.normalize = normalize
        self.dataset_kwargs = dataset_kwargs
        
        # Setup transforms
        self.transform = self._get_transforms()
    
    def _get_transforms(self):
        """Get image transforms."""
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ]
        
        if self.normalize:
            # Use ImageNet normalization by default
            # You might want to compute custom normalization for your dataset
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        return transforms.Compose(transform_list)
    
    def get_dataset(self):
        """Get the Many00 dataset."""
        return Many00Dataset(
            root_path=self.root_path,
            transform=self.transform,
            **self.dataset_kwargs
        )
    
    def get_dataloader(self):
        """Get the data loader."""
        dataset = self.get_dataset()
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        
        print(f"Many00 DataLoader created: {len(dataset)} samples, {len(dataloader)} batches")
        return dataloader
    
    def get_sample_batch(self):
        """Get a sample batch for testing."""
        dataloader = self.get_dataloader()
        return next(iter(dataloader))
    
    def analyze_dataset(self):
        """Analyze the dataset and print statistics."""
        dataset = self.get_dataset()
        
        print("\n=== Many00 Dataset Analysis ===")
        print(f"Total samples: {len(dataset)}")
        print(f"Number of classes: {len(dataset.classes)}")
        print(f"Classes: {dataset.classes}")
        
        # Class distribution
        class_dist = dataset.get_class_distribution()
        print("\nClass distribution:")
        for class_name, count in class_dist.items():
            print(f"  {class_name}: {count} samples")
        
        # Sample some images to check loading
        print(f"\nTesting image loading...")
        try:
            sample_image, sample_label = dataset[0]
            print(f"Sample image shape: {sample_image.shape}")
            print(f"Sample label: {sample_label}")
            print("Image loading successful!")
        except Exception as e:
            print(f"Error loading sample image: {e}")
        
        return dataset


# TEMPLATE FOR CONFIGURATION
"""
Example configuration for Many00 dataset:

# If your dataset structure is:
# many00/
#   ├── class1/
#   │   ├── img1.jpg
#   │   └── img2.jpg
#   └── class2/
#       ├── img3.jpg
#       └── img4.jpg

loader = Many00DataLoader(
    root_path='/path/to/many00',
    batch_size=32,
    recursive_search=False  # Images are directly in class folders
)

# If your dataset structure is:
# many00/
#   ├── images/
#   │   ├── img1.jpg
#   │   ├── img2.jpg
#   │   └── ...
#   └── annotations.csv  # Contains: image_path,label

loader = Many00DataLoader(
    root_path='/path/to/many00',
    batch_size=32,
    annotation_file='annotations.csv'
)

# If you want only specific classes:
loader = Many00DataLoader(
    root_path='/path/to/many00',
    batch_size=32,
    subset_classes=['class1', 'class3', 'class5'],
    max_samples=10000
)
"""