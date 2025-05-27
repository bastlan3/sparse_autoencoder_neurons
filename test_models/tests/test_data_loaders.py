import unittest
from unittest.mock import patch, MagicMock, call
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData

# Assuming the project structure allows this import path
# If not, sys.path adjustments might be needed in a real scenario,
# but here we assume the environment is set up for this.
from test_models.data.imagenet_loader import ImageNetDataLoader
from test_models.data.many00_loader import Many00DataLoader, Many00Dataset

class TestImageNetDataLoader(unittest.TestCase):

    @patch('torchvision.datasets.ImageFolder')
    def setUp(self, MockImageFolder):
        # Configure the mock to return a FakeData instance
        self.fake_data_instance = FakeData(size=10, image_size=(3, 224, 224), num_classes=2, transform=lambda x: x)
        MockImageFolder.return_value = self.fake_data_instance
        self.MockImageFolder = MockImageFolder

    def test_instantiation(self):
        """Test that ImageNetDataLoader instantiates correctly."""
        try:
            loader = ImageNetDataLoader(root_path='fake/imagenet', batch_size=4, num_workers=0)
            self.assertIsNotNone(loader)
        except Exception as e:
            self.fail(f"ImageNetDataLoader instantiation failed: {e}")

    def test_get_train_loader(self):
        """Test getting the training DataLoader."""
        loader = ImageNetDataLoader(root_path='fake/imagenet', batch_size=4, num_workers=0)
        train_dataloader = loader.get_train_loader()

        self.assertIsInstance(train_dataloader, DataLoader)
        self.assertIsInstance(train_dataloader.dataset, FakeData) # Check it's our FakeData
        # Check that ImageFolder was called for the 'train' split
        self.MockImageFolder.assert_any_call('fake/imagenet/train', transform=unittest.mock.ANY)


    def test_get_val_loader(self):
        """Test getting the validation DataLoader."""
        loader = ImageNetDataLoader(root_path='fake/imagenet', batch_size=4, num_workers=0)
        val_dataloader = loader.get_val_loader()

        self.assertIsInstance(val_dataloader, DataLoader)
        self.assertIsInstance(val_dataloader.dataset, FakeData) # Check it's our FakeData
        # Check that ImageFolder was called for the 'val' split
        self.MockImageFolder.assert_any_call('fake/imagenet/val', transform=unittest.mock.ANY)


class TestMany00DataLoader(unittest.TestCase):

    @patch('test_models.data.many00_loader.Many00Dataset._load_directory_based')
    def setUp(self, mock_load_directory_based):
        # This mock will prevent actual file system access during Many00Dataset initialization
        self.mock_load_directory_based = mock_load_directory_based
        
        # We need to ensure that the Many00Dataset instance created internally
        # has some minimal viable attributes after _load_dataset (which calls _load_directory_based)
        # We'll patch the instance that gets created.
        
        # Since Many00Dataset is instantiated within Many00DataLoader,
        # we can use a side_effect on the patch to modify the instance,
        # or patch its __init__ or _load_dataset more broadly.
        # For simplicity here, we let _load_directory_based be called,
        # but it does nothing. We'll then make get_dataset return a pre-configured FakeData.

        self.fake_many00_data = FakeData(size=5, image_size=(3, 224, 224), num_classes=1, transform=lambda x: x)

    # More targeted patch for Many00Dataset instantiation and its loading process
    @patch('test_models.data.many00_loader.Many00Dataset.__init__')
    def test_instantiation(self, MockMany00DatasetInit):
        """Test that Many00DataLoader instantiates correctly."""
        MockMany00DatasetInit.return_value = None # __init__ should return None
        
        # To make get_dataloader work, we need get_dataset to return something DataLoader-compatible
        # Let's make the Many00Dataset instance (mocked here) have necessary attributes
        # or mock the get_dataset method itself.

        try:
            # We pass dummy_kwarg to ensure it's passed down, not strictly necessary for this test
            loader = Many00DataLoader(root_path='fake/many00', batch_size=4, num_workers=0, dummy_kwarg='test_val')
            self.assertIsNotNone(loader)
            # Check that Many00Dataset was initialized with root_path and transform, and our dummy_kwarg
            MockMany00DatasetInit.assert_called_once_with(
                root_path='fake/many00',
                transform=unittest.mock.ANY,
                dummy_kwarg='test_val' # Check that kwargs are passed
            )
        except Exception as e:
            self.fail(f"Many00DataLoader instantiation failed: {e}")

    @patch('test_models.data.many00_loader.Many00Dataset')
    def test_get_dataloader(self, MockMany00Dataset):
        """Test getting the DataLoader from Many00DataLoader."""
        # Configure the mock Many00Dataset instance that will be returned by get_dataset
        mock_dataset_instance = self.fake_many00_data # Use FakeData as the dataset
        MockMany00Dataset.return_value = mock_dataset_instance

        loader = Many00DataLoader(root_path='fake/many00', batch_size=4, num_workers=0)
        
        # Override the internal get_dataset to return our mock that returns FakeData
        # This is a bit more direct for testing get_dataloader behavior
        loader.get_dataset = MagicMock(return_value=self.fake_many00_data)

        train_dataloader = loader.get_dataloader() # Corresponds to get_train_loader conceptually

        self.assertIsInstance(train_dataloader, DataLoader)
        self.assertIsInstance(train_dataloader.dataset, FakeData)
        loader.get_dataset.assert_called_once()


# New tests for Many00Dataset with real file loading
import tempfile
import shutil
import os
from PIL import Image
from torchvision import transforms as T # Use T to avoid conflict

def create_dummy_image(path, size=(64, 64), format="PNG", color='red'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new('RGB', size, color=color)
    img.save(path, format)

class TestMany00DatasetRealFiles(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # Create mock directory structure
        # category_A/img1.png, category_A/img2.jpg
        # category_B/img3.png
        create_dummy_image(os.path.join(self.temp_dir, "category_A", "img1.png"), format="PNG")
        create_dummy_image(os.path.join(self.temp_dir, "category_A", "img2.jpg"), format="JPEG")
        create_dummy_image(os.path.join(self.temp_dir, "category_B", "img3.png"), format="PNG")
        # Add a non-image file to ensure it's ignored
        with open(os.path.join(self.temp_dir, "category_A", "notes.txt"), "w") as f:
            f.write("test")
        # Add a nested directory to test recursive_search (default is True)
        create_dummy_image(os.path.join(self.temp_dir, "category_A", "nested", "img4.png"), format="PNG", color='blue')


    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_initialization_and_loading(self):
        """Test Many00Dataset loads data correctly from directory structure."""
        dataset = Many00Dataset(root_path=self.temp_dir)
        
        self.assertEqual(len(dataset.image_paths), 4) # img1, img2, img3, img4
        self.assertEqual(len(dataset.labels), 4)
        self.assertEqual(len(dataset.classes), 2)
        self.assertEqual(dataset.classes, ['category_A', 'category_B']) # Sorted
        self.assertEqual(dataset.class_to_idx, {'category_A': 0, 'category_B': 1})

        # Check paths and labels (order might vary due to os.walk, so check presence and corresponding label)
        expected_files = {
            os.path.join(self.temp_dir, "category_A", "img1.png"): 0,
            os.path.join(self.temp_dir, "category_A", "img2.jpg"): 0,
            os.path.join(self.temp_dir, "category_A", "nested", "img4.png"): 0,
            os.path.join(self.temp_dir, "category_B", "img3.png"): 1,
        }
        
        loaded_files_with_labels = {}
        for path, label in zip(dataset.image_paths, dataset.labels):
            loaded_files_with_labels[path] = label
        
        self.assertEqual(loaded_files_with_labels, expected_files)


    def test_len_method(self):
        """Test Many00Dataset __len__ method."""
        dataset = Many00Dataset(root_path=self.temp_dir)
        self.assertEqual(len(dataset), 4)

    def test_getitem_method(self):
        """Test Many00Dataset __getitem__ method with basic ToTensor transform."""
        transform = T.Compose([T.ToTensor()])
        dataset = Many00Dataset(root_path=self.temp_dir, transform=transform)
        
        # Assuming img1.png from category_A is one of the items
        # Find its index, as order isn't guaranteed if not sorting explicitly in _load_images_from_directory
        # Based on current implementation, os.walk is used, order can be tricky.
        # Let's find a specific image
        target_path = os.path.join(self.temp_dir, "category_A", "img1.png")
        item_idx = -1
        for i, path in enumerate(dataset.image_paths):
            if path == target_path:
                item_idx = i
                break
        self.assertNotEqual(item_idx, -1, "Target image not found in dataset paths")

        img_tensor, label = dataset[item_idx]
        
        self.assertIsInstance(img_tensor, torch.Tensor)
        self.assertEqual(label, 0) # category_A should be index 0
        self.assertEqual(img_tensor.shape, (3, 64, 64)) # Default dummy image size is 64x64

    def test_getitem_with_dataloader_transform(self):
        """Test Many00Dataset __getitem__ when used with Many00DataLoader's transforms."""
        # Many00DataLoader will apply its own transforms, including Resize
        loader = Many00DataLoader(root_path=self.temp_dir, image_size=32, batch_size=1)
        dataset = loader.get_dataset() # This dataset will have the loader's transform

        target_path = os.path.join(self.temp_dir, "category_B", "img3.png")
        item_idx = -1
        for i, path in enumerate(dataset.image_paths):
            if path == target_path:
                item_idx = i
                break
        self.assertNotEqual(item_idx, -1, "Target image not found in dataset paths")
        
        img_tensor, label = dataset[item_idx]
        
        self.assertIsInstance(img_tensor, torch.Tensor)
        self.assertEqual(label, 1) # category_B should be index 1
        # Shape should be (3, 32, 32) due to image_size=32 in Many00DataLoader
        # And normalization means values might not be 0-1.
        self.assertEqual(img_tensor.shape, (3, 32, 32))

    def test_non_recursive_search(self):
        """Test non-recursive search for images."""
        dataset = Many00Dataset(root_path=self.temp_dir, recursive_search=False)
        # Should only find img1.png, img2.jpg, img3.png (3 images), not img4.png in nested dir
        self.assertEqual(len(dataset.image_paths), 3)
        
        nested_img_path = os.path.join(self.temp_dir, "category_A", "nested", "img4.png")
        self.assertNotIn(nested_img_path, dataset.image_paths)


if __name__ == '__main__':
    unittest.main()
