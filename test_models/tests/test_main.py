import unittest
from unittest.mock import patch, MagicMock, mock_open
import torch

# Import the main function to be tested
from test_models.main import main

class TestMainIntegration(unittest.TestCase):

    @patch('test_models.main.torch.randperm') # For stable subset selection
    @patch('test_models.main.visualize_dictionary_features')
    @patch('test_models.main.save_checkpoint')
    @patch('test_models.main.ResNetFeatureExtractor')
    @patch('test_models.main.ImageNetDataLoader')
    @patch('test_models.main.Many00DataLoader') 
    @patch('test_models.main.wandb')
    @patch('builtins.open', new_callable=mock_open, read_data='{"resnet": {"model_name": "resnet50", "target_layer": "avgpool", "pretrained": false}, "sparse_dict": {"dict_size": 16, "sparsity_coef": 0.01}, "data": {"imagenet_path": "fake/path", "many00_path": "fake/path", "num_workers": 0}, "training": {"batch_size": 1, "num_epochs": 1, "learning_rate": 0.001, "log_interval": 1, "val_interval": 1, "viz_interval": 1}}')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_runs_minimal_epoch(
        self, 
        mock_parse_args, 
        mock_file_open, 
        mock_wandb, 
        MockMany00DataLoader, # Unused in this specific test flow directly for train/val
        MockImageNetDataLoader, 
        MockResNetFeatureExtractor,
        mock_save_checkpoint,
        mock_visualize_dictionary_features,
        mock_torch_randperm
    ):
        # 1. Configure argparse mock
        mock_parse_args.return_value = MagicMock(
            config='dummy_config.json', 
            experiment_name='test_run', 
            use_subset=True  # This is important for ResNet mock output shape
        )

        # 2. Configure ResNetFeatureExtractor mock
        # The main script will call resnet_extractor(dummy_input) to get feature_dim
        # Then it calls resnet_extractor(images) in the loop.
        # If use_subset is True, features are then subsampled.
        mock_resnet_instance = MockResNetFeatureExtractor.return_value
        # This is for the initial feature_dim detection call
        mock_resnet_instance.return_value = torch.randn(1, 2048) # Batch 1, Full ResNet features
        
        # 3. Configure ImageNetDataLoader mock
        dummy_images = torch.randn(1, 3, 224, 224) # Batch size 1 as per config
        dummy_labels = torch.randint(0, 1, (1,))
        
        mock_imagenet_loader_instance = MockImageNetDataLoader.return_value
        mock_imagenet_loader_instance.get_train_loader.return_value = [(dummy_images, dummy_labels)] # Iterable of one batch
        mock_imagenet_loader_instance.get_val_loader.return_value = [(dummy_images, dummy_labels)]   # Iterable of one batch

        # 4. Configure torch.randperm mock for stable feature selection in subset mode
        # If use_subset=True, features are features[:, indices]
        # where indices = torch.randperm(feature_dim)[:64]
        # So, the mocked ResNet output in the loop should be 2048, then it's sliced to 64.
        # The SparseDictionary input_dim will be 64.
        mock_torch_randperm.return_value = torch.arange(2048) # Return a deterministic permutation

        # Call the main function
        try:
            main()
        except Exception as e:
            self.fail(f"main() raised an exception: {e}")

        # Assertions
        mock_file_open.assert_called_once_with('dummy_config.json', 'r')
        mock_wandb.init.assert_called_once()

        # Check ResNet instantiation (optional, but good to ensure it's called as expected)
        # The config specifies resnet50 and avgpool. Pretrained should be False from config.
        MockResNetFeatureExtractor.assert_called_once_with(
            model_name='resnet50',
            layer_name='avgpool',
            pretrained=False # This comes from our mock config
        )
        
        # Check ResNet was called (at least once for feature dim, once for training batch)
        self.assertTrue(mock_resnet_instance.call_count >= 2)


        MockImageNetDataLoader.assert_called_once_with(
            root_path='fake/path', # From mock config
            batch_size=1,          # From mock config
            num_workers=0          # From mock config
        )
        mock_imagenet_loader_instance.get_train_loader.assert_called_once()
        mock_imagenet_loader_instance.get_val_loader.assert_called_once() # Called due to val_interval = 1

        # Check that torch.randperm was called because use_subset=True
        # It's called once for feature dim calculation (if that path uses it)
        # and once per training batch, and once per validation batch.
        # The dummy ResNet output for feature_dim is (1, 2048).
        # The randperm in main is: indices = torch.randperm(feature_dim)[:64]
        # So it should be called with 2048.
        # It is also called inside validate()
        self.assertTrue(mock_torch_randperm.call_count >= 2) # At least once in train, once in val
        mock_torch_randperm.assert_any_call(2048)


        # Check if save_checkpoint was called. Given val_interval=1, and assuming val_loss could be new best.
        # It's called if val_loss < best_loss (initially inf).
        mock_save_checkpoint.assert_called_once()

        # Check if visualize_dictionary_features was called, viz_interval = 1
        mock_visualize_dictionary_features.assert_called_once()
        
        self.assertTrue(mock_wandb.log.called)
        mock_wandb.finish.assert_called_once()

if __name__ == '__main__':
    unittest.main()
